import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

class GraphTopoAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 topo_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 residual=False,
                 concat=True,
                 last_layer=False):
        super(GraphTopoAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        # weight matrix Wl for leverage property
        if last_layer:
            self.fl = nn.Linear(in_dim+topo_dim, out_dim, bias=False)
        else:
            self.fl = nn.Linear(in_dim, num_heads*out_dim, bias=False)
        # weight matrix Wc for aggregation context
        self.fc = nn.Parameter(torch.Tensor(size=(in_dim+topo_dim, num_heads*out_dim)))
        # weight matrix Wq for neighbors' querying
        self.fq = nn.Parameter(torch.Tensor(size=(in_dim, num_heads*out_dim)))
        nn.init.xavier_normal_(self.fl.weight.data)
        nn.init.constant_(self.fc.data, 10e-3)
        nn.init.constant_(self.fq.data, 10e-3)
        self.attn_activation = nn.ELU()
        self.softmax = edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fl = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fl.weight.data)
            else:
                self.res_fl = None
        self.concat = concat
        self.last_layer = last_layer

    def forward(self, inputs, topo):
        # prepare
        h, t = self.feat_drop(inputs), self.feat_drop(topo)  # NxD, N*T
        if not self.last_layer:
            ft = self.fl(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            ft_c = torch.matmul(torch.cat((h, t), 1), self.fc).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            ft_q = torch.matmul(h, self.fq).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            self.g.ndata.update({'ft' : ft, 'ft_c' : ft_c, 'ft_q' : ft_q})
            self.g.apply_edges(self.edge_attention)
            self.edge_softmax()

            l_s = int(0.713*self.g.edata['a_drop'].shape[0])
            topk, _ = torch.topk(self.g.edata['a_drop'], l_s, largest=False, dim=0)
            thd = torch.squeeze(topk[-1])
            self.g.edata['a_drop'] = self.g.edata['a_drop'].squeeze()
            self.g.edata['a_drop'] = torch.where(self.g.edata['a_drop']-thd<0, self.g.edata['a_drop'].new([0.0]), self.g.edata['a_drop'])
            attn_ratio = torch.div((self.g.edata['a_drop'].sum(0).squeeze()+topk.sum(0).squeeze()), self.g.edata['a_drop'].sum(0).squeeze())
            self.g.edata['a_drop'] = self.g.edata['a_drop'] * attn_ratio
            self.g.edata['a_drop'] = self.g.edata['a_drop'].unsqueeze(-1)
            
            self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
            ret = self.g.ndata['ft']
            if self.residual:
                if self.res_fl is not None:
                    resval = self.res_fl(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
                else:
                    resval = torch.unsqueeze(h, 1)  # Nx1xD'
                ret = resval + ret
            ret = torch.cat((ret.flatten(1), ft.mean(1).squeeze()), 1) if self.concat else ret.flatten(1)
        else:
            ret = self.fl(torch.cat((h, t), 1))
        return ret

    def edge_attention(self, edges):
        c = edges.dst['ft_c']
        q = edges.src['ft_q'] - c
        a = (q * c).sum(-1).unsqueeze(-1)
        return {'a': self.attn_activation(a)}
        
    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        self.g.edata['a_drop'] = self.attn_drop(attention)

class GTN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 feats_d,
                 feats_t_d,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 residual,
                 concat):
        super(GTN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
            
        # input projection (no residual)
        self.gat_layers.append(GraphTopoAttention(g, feats_d, feats_t_d, num_hidden, heads[0], 
                                                feat_drop, attn_drop, False, concat))
        # hidden layers
        fix_d = concat*(feats_d)
        for l in range(1, num_layers+1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphTopoAttention(g, num_hidden*(heads[l-1]+1*concat), feats_t_d, 
                            num_hidden, heads[l], feat_drop, attn_drop, residual, concat))
        # output projection
        self.gat_layers.append(GraphTopoAttention(g, num_hidden*(heads[l-1]+1*concat), feats_t_d, 
                num_classes, heads[-1], feat_drop, attn_drop, residual, concat, True))

    def forward(self, inputs, topo):
        h, t = inputs, F.normalize(topo)
        for l in range(self.num_layers+1):
            h = self.gat_layers[l](h, t)
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](h, t)
        return logits
        