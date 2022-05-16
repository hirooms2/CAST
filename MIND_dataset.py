from MIND_corpus import MIND_Corpus
import time
import platform
from config import Config
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader


class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_body_text = corpus.news_body_text
        self.news_body_mask = corpus.news_body_mask
        self.news_mask = corpus.news_mask
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in
                              range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    def negative_sampling(self):
        print('\nBegin negative sampling, training sample num : %d' % self.num)
        start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4]
            news_num = len(negative_samples)
            if news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % news_num]
            else:
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
        end_time = time.time()
        print('End negative sampling, used time : %.3fs' % (end_time - start_time))

    # user_ID                       : [1]
    # user_category                 : [max_history_num]
    # user_subCategory              : [max_history_num]
    # user_title_text               : [max_history_num, max_title_length]
    # user_title_mask               : [max_history_num, max_title_length]
    # user_title_entity             : [max_history_num, max_title_length]
    # user_abstract_text            : [max_history_num, max_abstract_length]
    # user_abstract_mask            : [max_history_num, max_abstract_length]
    # user_abstract_entity          : [max_history_num, max_abstract_length]
    # user_body_text                : [max_history_num, max_body_length]
    # user_body_mask                : [max_history_num, max_body_length]
    # user_news_mask                : [max_history_num, self.max_title_length, self.max_body_length+self.max_title_length]
    # user_history_graph            : [max_history_num, max_history_num]
    # user_history_category_mask    : [category_num + 1]
    # user_history_category_indices : [max_history_num]
    # user_history_mask             : [max_history_num]
    # user_history_position         : [max_history_num]
    # news_category                 : [1 + negative_sample_num]
    # news_subCategory              : [1 + negative_sample_num]
    # news_title_text               : [1 + negative_sample_num, max_title_length]
    # news_title_mask               : [1 + negative_sample_num, max_title_length]
    # news_title_entity             : [1 + negative_sample_num, max_title_length]
    # news_abstract_text            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_mask            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_entity          : [1 + negative_sample_num, max_abstract_length]
    # news_body_text                : [1 + negative_sample_num, max_body_length]
    # news_body_mask                : [1 + negative_sample_num, max_body_length]
    # news_mask                     : [1 + negative_sample_num, self.max_title_length, self.max_body_length+self.max_title_length]
    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]

        return self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[
            history_index], self.news_title_mask[history_index], self.news_body_text[history_index], \
               self.news_body_mask[history_index], self.news_mask[history_index], train_behavior[2], train_behavior[6], \
               self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[
                   sample_index], self.news_title_mask[sample_index], self.news_body_text[sample_index], \
               self.news_body_mask[sample_index], self.news_mask[sample_index]

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_body_text = corpus.news_body_text
        self.news_body_mask = corpus.news_body_mask
        self.news_mask = corpus.news_mask
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

    # user_ID                       : [1]
    # user_category                 : [max_history_num]
    # user_subCategory              : [max_history_num]
    # user_title_text               : [max_history_num, max_title_length]
    # user_title_mask               : [max_history_num, max_title_length]
    # user_title_entity             : [max_history_num, max_title_length]
    # user_abstract_text            : [max_history_num, max_abstract_length]
    # user_abstract_mask            : [max_history_num, max_abstract_length]
    # user_abstract_entity          : [max_history_num, max_abstract_length]
    # user_body_text                : [max_history_num, max_body_length]
    # user_body_mask                : [max_history_num, max_body_length]
    # user_news_mask                : [max_history_num, self.max_title_length, self.max_body_length+self.max_title_length]
    # user_history_graph            : [max_history_num, max_history_num]
    # user_history_category_mask    : [category_num + 1]
    # user_history_category_indices : [max_history_num]
    # user_history_mask             : [max_history_num]
    # user_history_position         : [max_history_num]
    # news_category                 : [1]
    # news_subCategory              : [1]
    # news_title_text               : [max_title_length]
    # news_title_mask               : [max_title_length]
    # news_title_entity             : [max_title_length]
    # news_abstract_text            : [max_abstract_length]
    # news_abstract_mask            : [max_abstract_length]
    # news_abstract_entity          : [max_abstract_length]
    # news_body_text                : [max_body_length]
    # news_body_mask                : [max_body_length]
    # news_mask                     : [self.max_title_length, self.max_body_length+self.max_title_length]
    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]
        behavior_index = behavior[4]

        return self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[
            history_index], self.news_title_mask[history_index], self.news_body_text[history_index], \
               self.news_body_mask[history_index], self.news_mask[history_index], behavior[2], behavior[5], \
               self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], \
               self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], \
               self.news_body_text[candidate_news_index], self.news_body_mask[candidate_news_index], self.news_mask[
                   candidate_news_index]

    def __len__(self):
        return self.num
