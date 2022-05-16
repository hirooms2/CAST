import math
import numpy as np
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from newsEncoders import NewsEncoder


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.device = torch.device('cuda')
        self.auxiliary_loss = None

    # Input
    # user_title_text               : [batch_size, max_history_num, max_title_length]
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_body_text                : [batch_size, max_history_num, user_body_text]
    # user_body_mask                : [batch_size, max_history_num, user_body_mask]
    # user_category                 : [batch_size, max_history_num]
    # user_subCategory              : [batch_size, max_history_num]
    # user_mask                     : [batch_size, max_history_num, max_title_length, max_title_length + max_body_length]
    # user_history_mask             : [batch_size, max_history_num]
    # user_history_position         : [batch_size, max_history_num]
    # candidate_news_representaion  : [batch_size, news_num, news_embedding_dim]
    # Output
    # user_representation           : [batch_size, news_embedding_dim]

    def forward(self, user_title_text, user_title_mask, user_body_text, user_body_mask, user_category, user_subCategory, \
                user_mask, user_history_mask, user_history_position, candidate_news_representaion):
        raise Exception('Function forward must be implemented at sub-class')


class CAST(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(CAST, self).__init__(news_encoder, config)
        self.position_embedding = nn.Embedding(num_embeddings=config.max_history_num + 2,
                                               embedding_dim=config.position_dim)  # +2: padding (0), candidate_news (1)
        self.affine1 = nn.Linear(in_features=(self.news_embedding_dim + config.position_dim) * 2,
                                 out_features=config.attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=config.attention_dim, out_features=1, bias=True)
        self.position_dim = config.position_dim
        self.max_history_num = config.max_history_num

    def initialize(self):
        print('CAST initialize')
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.position_embedding.weight[0])
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, user_title_text, user_title_mask, user_body_text, user_body_mask, user_category, user_subCategory, \
                user_mask, user_history_mask, user_history_position, candidate_news_representaion):
        news_num = candidate_news_representaion.size(1)
        batch_size = candidate_news_representaion.size(0)
        user_history_position = user_history_position.long()  # [batch_size, max_history_num]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_body_text, user_body_mask,
                                              user_category, user_subCategory,
                                              user_mask)  # [batch_size, max_history_num, news_embedding_dim]

        pos = self.position_embedding(user_history_position)  # [batch_size, max_history_num, position_dim]
        history_embedding = torch.cat([history_embedding, pos],
                                      dim=2)  # [batch_size, max_history_num, news_embedding_dim + position_dim]
        can_pos = torch.ones((batch_size, news_num)).long().cuda(non_blocking=True)  # [batch_size, news_num]
        can_pos = self.position_embedding(can_pos)  # [batch_size, news_num, position_dim]
        candidate_news_representaion = torch.cat([candidate_news_representaion, can_pos],
                                                 dim=2)  # [batch_size, news_num, news_embedding_dim + position_dim]

        user_history_mask = user_history_mask.unsqueeze(dim=1).expand(-1, news_num,
                                                                      -1)  # [batch_size, news_num, max_history_num]
        candidate_news_representaion = candidate_news_representaion.unsqueeze(dim=2).expand(-1, -1,
                                                                                            self.max_history_num,
                                                                                            -1)  # [batch_size, news_num, max_history_num, news_embedding_dim + position_dim]
        history_embedding = history_embedding.unsqueeze(dim=1).expand(-1, news_num, -1,
                                                                      -1)  # [batch_size, news_num, max_history_num, news_embedding_dim + position_dim]
        concat_embeddings = torch.cat([candidate_news_representaion, history_embedding],
                                      dim=3)  # [batch_size, news_num, max_history_num, (news_embedding_dim + position_dim) * 2]
        hidden = F.relu(self.affine1(concat_embeddings),
                        inplace=True)  # [batch_size, news_num, max_history_num, attention_dim]
        a = self.affine2(hidden).squeeze(dim=3)  # [batch_size, news_num, max_history_num]
        alpha = F.softmax(a.masked_fill(user_history_mask == 0, -1e9), dim=2)  # [batch_size, news_num, max_history_num]
        user_representation = (alpha.unsqueeze(dim=3) * history_embedding).sum(dim=2,
                                                                               keepdim=False)  # [batch_size, news_num, news_embedding_dim + position_dim]
        candidate_news_representaion = candidate_news_representaion[:, :, 0, :]
        return user_representation, candidate_news_representaion
