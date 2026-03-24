import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 核心模块 1：基于协方差的特征正交解耦损失 (IT-MoE)
# ==========================================
class Orthogonal_Decoupling_Loss(nn.Module):
    """
    基于协方差/正交性的解耦损失，严格计算 Shared 和 Specific 特征之间的正交性。
    该损失无内部参数，严格为正，完全杜绝了优化过程中的 NaN 崩塌现象。
    """
    def __init__(self):
        super(Orthogonal_Decoupling_Loss, self).__init__()

    def forward(self, x_shared, x_specific):
        x_shared = x_shared.view(-1, x_shared.size(-1))
        x_specific = x_specific.view(-1, x_specific.size(-1))

        # L2 归一化
        x_shared_norm = F.normalize(x_shared, p=2, dim=1)
        x_specific_norm = F.normalize(x_specific, p=2, dim=1)

        # 惩罚相关性矩阵的非对角线
        correlation_matrix = torch.matmul(x_shared_norm.t(), x_specific_norm)
        loss = torch.norm(correlation_matrix, p='fro') / x_shared.size(0)
        return loss

class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='none')

    def forward(self, log_pred, target, mask, u_student=None, gamma=1.0):
        mask_ = mask.view(-1, 1)
        loss_elements = self.loss(log_pred * mask_, target * mask_)
        loss_per_sample = loss_elements.sum(dim=-1)
        
        # 【终极修复核心】
        if u_student is not None:
            # 1. 必须 detach()! 严禁梯度通过权重传导，防止模型为降Loss故意输出高不确定性作弊
            u_detached = u_student.detach().squeeze(-1)
            
            # 2. 放弃容易饿死模型的 (1 - u)，改用理论更优雅、永远大于0的指数惩罚 exp(-gamma * u)
            # 或者使用保守线性截断： torch.clamp(1.0 - u_detached, min=0.2)
            # 这里推荐使用带底线的平滑缩放，保证至少有 20% 的蒸馏知识流向学生
            weight = torch.clamp(1.0 - u_detached, min=0.2) 
            
            loss_per_sample = loss_per_sample * weight
            
        loss = torch.sum(loss_per_sample) / torch.sum(mask)   
        return loss

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

# ==========================================
# 核心模块 3：拓扑感知关系路由器 (TARR) + IT-MoE 层
# ==========================================
class IT_TARR_MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, dropout=0.1):
        super(IT_TARR_MoELayer, self).__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        self.num_specific = num_experts // 2
        self.num_shared = num_experts - self.num_specific
        self.experts = nn.ModuleList([PositionwiseFeedForward(d_model, d_ff, dropout) for _ in range(num_experts)])
        
        self.router_q = nn.Linear(d_model, d_model)
        self.router_k = nn.Linear(d_model, d_model)
        self.router_v = nn.Linear(d_model, d_model)
        self.w_dist = nn.Parameter(torch.tensor(0.1)) 
        self.w_spk = nn.Parameter(torch.tensor(1.0)) 
        self.router_out = nn.Linear(d_model, num_experts)
        
        self.vib_mu = nn.Linear(d_model, d_model)
        self.vib_logvar = nn.Linear(d_model, d_model)
        
        self.ortho_loss = Orthogonal_Decoupling_Loss()

    def forward(self, x, spk_idx):
        B, L, D = x.size()
        
        # 1. 注入拓扑图先验 (时序距离 + 说话人身份)
        dist_mat = torch.abs(torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)).to(x.device).float()
        dist_mat = dist_mat.unsqueeze(0).expand(B, L, L)
        spk_mat = (spk_idx.unsqueeze(1) == spk_idx.unsqueeze(2)).float()
        phi = self.w_dist * (-dist_mat) + self.w_spk * spk_mat
        
        q = self.router_q(x)
        k = self.router_k(x)
        v = self.router_v(x)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        attn_scores = attn_scores + phi 
        attn_weights = F.softmax(attn_scores, dim=-1)
        ctx = torch.matmul(attn_weights, v)
        
        gate_logits = self.router_out(x + ctx)
        expert_weights = F.softmax(gate_logits, dim=-1)
        
        out = torch.zeros_like(x)
        specific_out = torch.zeros_like(x)
        shared_out = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            e_out = expert(x)
            weighted_e_out = expert_weights[:, :, i].unsqueeze(-1) * e_out
            out += weighted_e_out
            if i < self.num_specific:
                specific_out += weighted_e_out
            else:
                shared_out += weighted_e_out
                
        # 2. 变分信息瓶颈 (VIB) 正则化
        mu = self.vib_mu(shared_out)
        logvar = self.vib_logvar(shared_out)
        logvar = torch.clamp(logvar, min=-15.0, max=10.0) # 截断防止指数爆炸
        
        z_shared = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        vib_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) / (B * L)
        
        # 3. 计算正交解耦惩罚
        mi_loss = self.ortho_loss(z_shared, specific_out)
        
        # 负载均衡防止路由崩塌
        importance = expert_weights.sum(dim=(0, 1))
        importance = importance / (importance.sum() + 1e-9)
        moe_balance_loss = self.num_experts * torch.sum(importance ** 2) - 1
        
        return out, moe_balance_loss, vib_loss, mi_loss

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x

class MoEEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, num_experts, dropout):
        super(MoEEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.moe_layer = IT_TARR_MoELayer(d_model, d_ff, num_experts, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask, spk_idx):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)
        
        out = self.dropout(context) + inputs_b
        moe_out, moe_balance_loss, vib_loss, mi_loss = self.moe_layer(out, spk_idx)
        return moe_out, moe_balance_loss, vib_loss, mi_loss

class MoEEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, num_experts=4, dropout=0.1):
        super(MoEEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.moe_inter = nn.ModuleList([MoEEncoderLayer(d_model, heads, d_ff, num_experts, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb, spk_idx):
        t_balance, t_vib, t_mi = 0.0, 0.0, 0.0
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b, bal, vib, mi = self.moe_inter[i](i, x_b, x_b, mask.eq(0), spk_idx)
                t_balance += bal; t_vib += vib; t_mi += mi
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b, bal, vib, mi = self.moe_inter[i](i, x_a, x_b, mask.eq(0), spk_idx)
                t_balance += bal; t_vib += vib; t_mi += mi
        return x_b, t_balance, t_vib, t_mi

class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        utters = torch.cat([a.unsqueeze(-2), b.unsqueeze(-2), c.unsqueeze(-2)], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        return torch.sum(utters_softmax * utters, dim=-2, keepdim=False)

class SDT_MoER_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, num_experts=4):
        super(SDT_MoER_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        padding_idx = 2 if n_speakers == 2 else 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        self.t_t = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        self.a_t = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        self.v_t = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)

        self.a_a = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        self.t_a = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        self.v_a = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)

        self.v_v = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        self.t_v = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        self.a_v = MoEEncoder(hidden_dim, hidden_dim, n_head, 1, num_experts, dropout)
        
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        self.t_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
        self.a_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
        self.v_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx.clone()
        if self.n_speakers in [2, 9]:
            pad = self.n_speakers
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = pad
        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        def run_encoder(enc, x_a, x_b):
            return enc(x_a, x_b, u_mask, spk_embeddings, spk_idx)

        t_t_out, bal1, vib1, mi1 = run_encoder(self.t_t, textf, textf)
        a_t_out, bal2, vib2, mi2 = run_encoder(self.a_t, acouf, textf)
        v_t_out, bal3, vib3, mi3 = run_encoder(self.v_t, visuf, textf)

        a_a_out, bal4, vib4, mi4 = run_encoder(self.a_a, acouf, acouf)
        t_a_out, bal5, vib5, mi5 = run_encoder(self.t_a, textf, acouf)
        v_a_out, bal6, vib6, mi6 = run_encoder(self.v_a, visuf, acouf)

        v_v_out, bal7, vib7, mi7 = run_encoder(self.v_v, visuf, visuf)
        t_v_out, bal8, vib8, mi8 = run_encoder(self.t_v, textf, visuf)
        a_v_out, bal9, vib9, mi9 = run_encoder(self.a_v, acouf, visuf)

        total_balance = bal1+bal2+bal3+bal4+bal5+bal6+bal7+bal8+bal9
        total_vib = vib1+vib2+vib3+vib4+vib5+vib6+vib7+vib8+vib9
        total_mi = mi1+mi2+mi3+mi4+mi5+mi6+mi7+mi8+mi9

        t_t_out = self.t_t_gate(t_t_out)
        a_t_out = self.a_t_gate(a_t_out)
        v_t_out = self.v_t_gate(v_t_out)
        a_a_out = self.a_a_gate(a_a_out)
        t_a_out = self.t_a_gate(t_a_out)
        v_a_out = self.v_a_gate(v_a_out)
        v_v_out = self.v_v_gate(v_v_out)
        t_v_out = self.t_v_gate(t_v_out)
        a_v_out = self.a_v_gate(a_v_out)

        t_transformer_out = self.features_reduce_t(torch.cat([t_t_out, a_t_out, v_t_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_out, t_a_out, v_a_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_out, t_v_out, a_v_out], dim=-1))

        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)

        # 1. 主分类输出
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)

        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)
        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        # 2. 温度平滑软标签（用于高质量蒸馏）
        kl_t_log_prob = F.log_softmax(t_final_out / self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out / self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out / self.temp, 2)
        kl_all_prob = F.softmax(all_final_out / self.temp, 2)

        # 3. 计算认知不确定性 (Evidence -> Alpha -> Uncertainty)
        t_alpha = F.softplus(t_final_out) + 1
        a_alpha = F.softplus(a_final_out) + 1
        v_alpha = F.softplus(v_final_out) + 1
        
        t_u = self.n_classes / torch.sum(t_alpha, dim=-1, keepdim=True)
        a_u = self.n_classes / torch.sum(a_alpha, dim=-1, keepdim=True)
        v_u = self.n_classes / torch.sum(v_alpha, dim=-1, keepdim=True)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob, \
               t_u, a_u, v_u, total_balance, total_vib, total_mi