import torch
import torch.nn.functional as F
import random
import numpy as np
import time
import dgl
from dgl import DGLGraph
import argparse
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils import load_data
from models import GTN

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def evaluate(g, feats, feats_t, labels, mask, model, loss_fcn, device):
    with torch.no_grad():
        model.eval()
        model.g = g
        for layer in model.gtn_layers:
            layer.g = g
        output = model(feats.float(), feats_t.float())
        loss = loss_fcn(output[mask], labels[mask])
        # predict = np.where(output[mask].data.cpu().numpy() >= 0.5, 1, 0)
        predict = np.argmax(output[mask].data.cpu().numpy(), axis=1)
        true = np.argmax(labels[mask].data.cpu().numpy(), axis=1)
        score = f1_score(true, predict, average='micro')
        return score, loss.item()

def train_main(args):
    # cpu or gpu
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    # create the dataset
    g, train_subgraphs, train_subfeats, train_subfeats_t, train_sublabels, train_submasks, val_subgraphs, val_subfeats, \
    val_subfeats_t, val_sublabels, val_submasks, test_subgraphs, test_subfeats, test_subfeats_t, test_sublabels, \
    test_submasks = load_data(args.prefix, args.num_layers, args.batch_size, args.concat, args.sample_number)

    # define the model and optimizer
    heads = ([args.num_heads] * (args.num_layers + 1))
    model = GTN(g,
                args.num_layers,
                train_subfeats[0].shape[1],
                args.sample_number*2,
                args.num_hidden,
                train_sublabels[0].shape[1],
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.residual,
                args.concat)
    model = model.to(device)
    attn_params_name = ['fc', 'fq']
    attn_params = []
    for p in attn_params_name:
        attn_params = attn_params + list(filter(lambda kv: p in kv[0], model.named_parameters()))
    base_params = [param[1] for param in model.named_parameters() if param not in attn_params]
    attn_params = [param[1] for param in attn_params]
    optimizer = torch.optim.Adam([{'params': base_params},
                                {'params': attn_params, 'lr': args.lr/10}],
                                lr=args.lr, weight_decay=args.weight_decay)

    loss_fcn = torch.nn.BCEWithLogitsLoss()

    # start training
    best_score, best_loss, cur_step = 0, 1000, 0
    for epoch in range(args.epochs):
        model.train()
        loss_list = []
        # shuffle
        idx = [i for i in range(len(train_subgraphs))]
        random.shuffle(idx)
        for i in range(len(train_subgraphs)):
            feats = torch.FloatTensor(train_subfeats[idx[i]]).to(device)
            feats_t = torch.FloatTensor(train_subfeats_t[idx[i]]).to(device)
            labels = torch.FloatTensor(train_sublabels[idx[i]]).to(device)
            model.g = train_subgraphs[idx[i]]
            for layer in model.gtn_layers:
                layer.g = train_subgraphs[idx[i]]
            output = model(feats.float(), feats_t.float())
            loss = loss_fcn(output[train_submasks[idx[i]]], labels[train_submasks[idx[i]]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        # validation
        score_list = []
        val_loss_list = []
        for i in range(len(val_subgraphs)):
            feats = torch.FloatTensor(val_subfeats[i]).to(device)
            feats_t = torch.FloatTensor(val_subfeats_t[i]).to(device)
            labels = torch.FloatTensor(val_sublabels[i]).to(device)
            score, val_loss = evaluate(val_subgraphs[i], 
                                        feats, 
                                        feats_t,
                                        labels, 
                                        val_submasks[i], 
                                        model, 
                                        loss_fcn,
                                        device)
            score_list.append(score)
            val_loss_list.append(val_loss)
        
        # early stop
        if sum(score_list)/len(score_list) > best_score: #or sum(val_loss_list)/len(val_loss_list) < best_loss:
            print("Epoch {:05d} |Train Loss: {:.4f} | Val Loss: {:.4f} | F1-Score: {:.4f} | save".format(epoch + 1, 
                                                                        sum(loss_list)/len(loss_list), 
                                                                        sum(val_loss_list)/len(val_loss_list),
                                                                        sum(score_list)/len(score_list)))
            torch.save(model.state_dict(), args.prefix.split('/')[-1]+'_best.pkl')
            best_score = sum(score_list)/len(score_list)
            best_loss = sum(val_loss_list)/len(val_loss_list)
            cur_step = 0
            optimizer.param_groups[0]['lr'] = args.lr
            optimizer.param_groups[-1]['lr'] = args.lr/10
        else:
            print("Epoch {:05d} |Train Loss: {:.4f} | Val Loss: {:.4f} | F1-Score: {:.4f} ".format(epoch + 1, 
                                                                        sum(loss_list)/len(loss_list), 
                                                                        sum(val_loss_list)/len(val_loss_list),
                                                                        sum(score_list)/len(score_list)))
            cur_step += 1
            if cur_step == int(args.patience/2):
                optimizer.param_groups[0]['lr'] = args.lr
                optimizer.param_groups[-1]['lr'] = args.lr
            if cur_step > args.patience:
                break
    # test
    model.load_state_dict(torch.load(args.prefix.split('/')[-1]+'_best.pkl'))
    test_score_list = []
    test_loss_list = []
    for i in range(len(test_subgraphs)):
        feats = torch.FloatTensor(test_subfeats[i]).to(device)
        feats_t = torch.FloatTensor(test_subfeats_t[i]).to(device)
        labels = torch.FloatTensor(test_sublabels[i]).to(device)
        test_score, test_loss = evaluate(test_subgraphs[i], 
                                feats, 
                                feats_t,
                                labels, 
                                test_submasks[i], 
                                model, 
                                loss_fcn,
                                device)
    test_score_list.append(test_score)
    test_loss_list.append(test_loss)
    print("The test Loss: {:.4f}, F1-Score: {:.4f}".format(sum(test_loss_list)/len(test_loss_list), 
                                                                sum(test_score_list)/len(test_score_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--concat", action="store_true", default=True,
                        help="concat neighbors with self")
    parser.add_argument("--in-drop", type=float, default=0.3,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help="weight decay")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=100,
                        help="used for early stop")
    parser.add_argument("--sample-number", type=int, default=32,
                        help="characteristic function sample number, delete feats_t.npy before change")
    parser.add_argument('--prefix', type=str, default='./data/amazon/amazon',
                        help="which dataset to use")
    args = parser.parse_args()
    print(args)
    train_main(args)
