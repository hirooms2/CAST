from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import newsEncoders
import userEncoders


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.news_encoder == 'CAST':
            self.news_encoder = newsEncoders.CAST(config)
        else:
            raise Exception(config.news_encoder + 'is not implemented')

        if config.user_encoder == 'CAST':
            self.user_encoder = userEncoders.CAST(self.news_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')
        

        self.model_name = config.news_encoder + '-' + config.user_encoder
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()

                                
    def forward(self, user_category, user_subCategory, user_title_text, user_title_mask, user_body_text, user_body_mask, user_news_mask, user_history_mask, user_history_position,\
                    news_category, news_subCategory, news_title_text, news_title_mask, news_body_text, news_body_mask, news_mask):
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_body_text, news_body_mask, news_category, news_subCategory, news_mask) # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        user_representation, news_representation = self.user_encoder(user_title_text, user_title_mask, user_body_text, user_body_mask, user_category, user_subCategory, \
                                            user_news_mask, user_history_mask, user_history_position, news_representation)                              # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        logits = (user_representation * news_representation).sum(dim=2) # dot-product
        
        return logits