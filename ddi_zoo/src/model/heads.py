from torch import nn
from fairseq import utils
import torch
import torch.nn.functional as F
import numpy as np


class BinaryClassMLPHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes
        self.rel_emb = nn.Embedding(self.num_classes, inner_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        
        self.dense = nn.Linear(input_dim * 2, inner_dim)
        self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        
    def forward(self, heads, tails, rels):

        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)

        x = torch.cat([heads, tails], dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(rels, x.unsqueeze(-1))

        return scores


class BinaryClassFeatIntegrationHead(nn.Module):
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()

        self.num_classes = num_classes
        self.rel_emb = nn.Embedding(self.num_classes, inner_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        
        self.dense = nn.Linear(input_dim * 2, inner_dim)
        self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        
        self.transMatrix = nn.Linear(input_dim, inner_dim)
        self.res_linear_head = nn.Linear(inner_dim * 3, input_dim)
        self.res_linear_tail = nn.Linear(inner_dim * 3, input_dim)

        self.res_linear_rel = nn.Linear(inner_dim * 3, inner_dim)


    def forward(self, heads, tails, rels):
        
        rels_ori = self.rel_emb(rels)
        
        input = torch.cat([self.transMatrix(heads), self.transMatrix(tails), rels_ori.squeeze(1)], dim=-1)
        delta_rel = self.dropout(self.activation_fn(self.res_linear_rel(input))).unsqueeze(1)

        rels = rels_ori + delta_rel 
        
        delta_t = self.dropout(self.activation_fn(self.res_linear_tail(input)))
        delta_h = self.dropout(self.activation_fn(self.res_linear_head(input)))
        
        heads_new = heads + delta_h 
        tails_new = tails + delta_t 
        
        x = torch.cat([heads_new, tails_new], dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(rels, x.unsqueeze(-1))
        
        return scores

    def forward_deltaT(self, heads, tails, rels):
        
        rels_ori = self.rel_emb(rels)

        input = torch.cat([self.transMatrix(heads), self.transMatrix(tails), rels_ori.squeeze(1)], dim=-1)
        delta_rel = self.dropout(self.activation_fn(self.res_linear_rel(input))).unsqueeze(1)

        rels = rels_ori + delta_rel

        delta_t = self.dropout(self.activation_fn(self.res_linear_tail(input)))
        
        heads_new = heads
        tails_new = tails + delta_t
        
        x = torch.cat([heads_new, tails_new], dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(rels, x.unsqueeze(-1))
        
        return scores

    def forward_deltaH(self, heads, tails, rels):
        
        rels_ori = self.rel_emb(rels)

        input = torch.cat([self.transMatrix(heads), self.transMatrix(tails), rels_ori.squeeze(1)], dim=-1)
        delta_rel = self.dropout(self.activation_fn(self.res_linear_rel(input))).unsqueeze(1)

        rels = rels_ori + delta_rel

        delta_h = self.dropout(self.activation_fn(self.res_linear_head(input)))
        
        heads_new = heads + delta_h
        tails_new = tails
        
        x = torch.cat([heads_new, tails_new], dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(rels, x.unsqueeze(-1))

        return scores

    def forward_NoDelta(self, heads, tails, rels):
        
        rels_ori = self.rel_emb(rels)

        rels = rels_ori

        heads_new = heads
        tails_new = tails
        
        x = torch.cat([heads_new, tails_new], dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(rels, x.unsqueeze(-1))

        return scores
