import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Attention, Context_Aware_Att


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(
                config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(
                config.max_body_length) + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num,
                                               embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num,
                                                  embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.auxiliary_loss = None

    def initialize(self):
        print('news encoder initialize')
        # nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # body_text           : [batch_size, news_num, max_body_length]
    # body_mask           : [batch_size, news_num, max_body_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # mask                : [batch_size, news_num, max_title_length + max_body_length]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, body_text, body_mask, category, subCategory, mask):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)  # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(
            subCategory)  # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat(
            [news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)],
            dim=2)  # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CAST(NewsEncoder):
    def __init__(self, config: Config):
        super(CAST, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_body_length = config.max_body_length
        self.word_embedding_dim = config.word_embedding_dim

        self.title_dim = config.head_num * config.head_dim
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim

        self.cast = Context_Aware_Att(config.head_num, config.head_dim, self.word_embedding_dim, self.max_title_length,
                                      self.max_body_length)
        self.attention = Attention(self.title_dim, config.attention_dim)

    def initialize(self):
        print('cast initialize')
        super().initialize()
        self.cast.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, body_text, body_mask, category, subCategory, mask):
        title_text = title_text.long()
        body_text = body_text.long()
        category = category.long()
        subCategory = subCategory.long()

        batch_size = category.size(0)
        news_num = category.size(1)

        title_mask = title_mask.view(
            [batch_size * news_num, self.max_title_length])  # [batch_size * news_num, max_title_length]
        body_mask = body_mask.view([batch_size * news_num, self.max_body_length])

        title = self.dropout(self.word_embedding(title_text)).view([batch_size * news_num, self.max_title_length,
                                                                    self.word_embedding_dim])  # [batch_size * news_num, max_title_length, word_embedding_dim]
        body = self.dropout(self.word_embedding(body_text)).view([batch_size * news_num, self.max_body_length,
                                                                  self.word_embedding_dim])  # [batch_size * news_num, max_content_length, word_embedding_dim]

        context_title = self.dropout(self.cast([title, body, body], mask))  # [batch_size * news_num, title_length, 400]

        title_rep = self.attention(context_title, title_mask).view(
            [batch_size, news_num, self.title_dim])  # [batch_size, news_num, 400]

        title_rep = self.feature_fusion(title_rep, category, subCategory)  # [batch_size, news_num, 500]

        return title_rep
