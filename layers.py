import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Context_Aware_Att(nn.Module):
    def __init__(self, nb_head: int, size_per_head: int, d_model: int, len_q: int, len_k: int, **kwargs):
        super(Context_Aware_Att, self).__init__(**kwargs)
        
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head

        self.len_q = len_q
        self.len_k = len_k

        self.attention_scalar = math.sqrt(float(self.size_per_head))
        
        self.W_Q = nn.Linear(in_features=d_model, out_features=self.output_dim, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=self.output_dim, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.output_dim, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, dim]
    # K    : [batch_size, len_k, dim]
    # V    : [batch_size, len_k, dim]
    # mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, nb_head * size_per_head]
    def forward(self, x, mask):
        Q_seq,K_seq,V_seq = x
        batch_size = Q_seq.size(0)

        ## [batch_size, body_length, dim] -> [batch_size, title_length + body_length, dim]
        K_seq = torch.cat([K_seq, Q_seq], 1)
        V_seq = torch.cat([V_seq, Q_seq], 1)

        ## [batch_size, title_length, title_length + body_length] -> [batch_size * nb_head, title_length, title_length + body_length]
        mask = mask.view([batch_size, 1, self.len_q, self.len_q+self.len_k]).repeat(1, self.nb_head, 1, 1).view([batch_size * self.nb_head, self.len_q, self.len_q+self.len_k])

        Q_seq = self.W_Q(Q_seq).view([batch_size, self.len_q, self.nb_head, self.size_per_head])
        K_seq = self.W_K(K_seq).view([batch_size, self.len_q+self.len_k, self.nb_head, self.size_per_head])
        V_seq = self.W_V(V_seq).view([batch_size, self.len_q+self.len_k, self.nb_head, self.size_per_head])
        Q_seq = Q_seq.permute(0, 2, 1, 3).contiguous().view([batch_size * self.nb_head, self.len_q, self.size_per_head])
        K_seq = K_seq.permute(0, 2, 1, 3).contiguous().view([batch_size * self.nb_head, self.len_q+self.len_k, self.size_per_head])
        V_seq = V_seq.permute(0, 2, 1, 3).contiguous().view([batch_size * self.nb_head, self.len_q+self.len_k, self.size_per_head])
        A = torch.bmm(Q_seq, K_seq.permute(0, 2, 1).contiguous()) / self.attention_scalar # [batch_size * nb_head, title_length, title_length + body_length]

        alpha = F.softmax(A.masked_fill(mask == 0, -1e9), dim=2)        # [batch_size * nb_head, title_length, title_length + body_length]

        out = torch.bmm(alpha, V_seq).view([batch_size, self.nb_head, self.len_q, self.size_per_head])             # [batch_size, nb_head, title_length, size_per_head]
        out = out.permute([0, 2, 1, 3]).contiguous().view([batch_size, self.len_q, self.output_dim])                  # [batch_size, title_length, nb_head * size_per_head]
        return out

class Attention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out