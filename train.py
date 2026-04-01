import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, MoER_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('/data/zzb/BaseLine/Nine/data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = MELDDataset('/data/zzb/BaseLine/Nine/data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() if args.cuda else d for d in data[:-1]]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, \
        t_u, a_u, v_u, moe_balance_loss, vib_loss, mi_loss = model(
            textf, visuf, acouf, umask, qmask, lengths)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

        t_u_flat = t_u.view(-1, 1)
        a_u_flat = a_u.view(-1, 1)
        v_u_flat = v_u.view(-1, 1)

        loss_cls = gamma_1 * loss_function(lp_all, labels_, umask) + \
                   gamma_2 * (loss_function(lp_1, labels_, umask) + 
                              loss_function(lp_2, labels_, umask) + 
                              loss_function(lp_3, labels_, umask))

        loss_distill = gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask, u_student=t_u_flat) + 
                                  kl_loss(kl_lp_2, kl_p_all, umask, u_student=a_u_flat) + 
                                  kl_loss(kl_lp_3, kl_p_all, umask, u_student=v_u_flat))

        loss = loss_cls + loss_distill + \
               0.0001 * moe_balance_loss + 0.0000001 * vib_loss + 0.001 * mi_loss

        lp_ = all_prob.view(-1, all_prob.size()[2])
        pred_ = torch.argmax(lp_, 1)
        
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.000005, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--num_experts', type=int, default=4, metavar='num_experts', help='number of MoE experts')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=8, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='MELD', help='dataset to train and test')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    model = MoER_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                           n_classes=n_classes,
                           hidden_dim=args.hidden_dim,
                           n_speakers=n_speakers,
                           dropout=args.dropout,
                           num_experts=args.num_experts)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))

    if args.cuda:
        model.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    kl_loss = MaskedKLDivLoss()

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0, batch_size=batch_size, num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        cw = torch.FloatTensor([1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668])
        loss_function = MaskedNLLLoss(cw.cuda() if args.cuda else cw)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0, batch_size=batch_size, num_workers=0)

    best_fscore, best_label, best_pred, best_mask = None, None, None, None
    all_fscore = []

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, kl_loss, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(
            model, loss_function, kl_loss, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(
            model, loss_function, kl_loss, test_loader, e)
        
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        
        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))

    print('Test performance..')
    max_fscore = max(all_fscore)
    print('F-Score: {}'.format(max_fscore))
    print('F-Score-index: {}'.format(all_fscore.index(max_fscore) + 1))
    
    save_dir = "/data/zzb/BaseLine/nine/main/result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    file_name = "record_{}_{}_{}_{}.pk".format(max_fscore, today.year, today.month, today.day)
    full_path = os.path.join(save_dir, file_name)
    
    if not os.path.exists(full_path):
        with open(full_path, 'wb') as f:
            pk.dump({}, f)
            
    with open(full_path, 'rb') as f:
        record = pk.load(f)
        
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max_fscore)
    else:
        record[key_] = [max_fscore]
        
    if record.get(key_+'record', False):
        record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    else:
        record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
        
    with open(full_path, 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
    
