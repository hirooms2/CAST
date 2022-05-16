import os
import json
import pickle
import collections
import re
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe
from config import Config
import torch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


pat = re.compile(r"[\w]+|[.,!?;|]")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


class MIND_Corpus:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'user_ID.json'
        news_ID_file = 'news_ID.json'
        category_file = 'category.json'
        subCategory_file = 'subCategory.json'
        vocabulary_file = 'vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(
            config.max_title_length) + '-' + str(config.max_body_length) + '.json'
        word_embedding_file = 'word_embedding-' + str(config.word_threshold) + '-' + str(
            config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(
            config.max_body_length) + '.pkl'
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file,
                                   word_embedding_file]

        if not all(list(map(lambda x: os.path.exists(x), preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            news_category_dict = {}

            # 1. user ID dictionay
            with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
                with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                    json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news_with_body.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities, body = line.split(
                            '\t')
                        if news_ID not in news_ID_dict:
                            news_ID_dict[news_ID] = len(news_ID_dict)
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)
                            # words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(
                            #     title.lower())

                            word_id = tokenizer(title.lower(), return_tensors="np", max_length=512).input_ids
                            words = tokenizer.convert_ids_to_tokens(word_id.squeeze(), skip_special_tokens=True)

                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:  # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:  # already appeared in training set
                                            word_counter[word] += 1
                            # words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(
                            #     abstract.lower())
                            word_id = tokenizer(abstract.lower(), return_tensors="np", max_length=512).input_ids
                            words = tokenizer.convert_ids_to_tokens(word_id.squeeze(), skip_special_tokens=True)

                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:  # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:  # already appeared in training set
                                            word_counter[word] += 1
                            # words = pat.findall(body.lower()) if config.tokenizer == 'MIND' else word_tokenize(
                            #     body.lower())
                            word_id = tokenizer(body.lower(), return_tensors="np", max_length=512).input_ids
                            words = tokenizer.convert_ids_to_tokens(word_id.squeeze(), skip_special_tokens=True)

                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:  # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:  # already appeared in training set
                                            word_counter[word] += 1
                        news_category_dict[news_ID] = category_dict[category]
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True)  # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Glove word embedding
            if config.word_embedding_dim == 300:
                glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
            else:
                glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../glove', max_vectors=10000000000)
            glove_stoi = glove.stoi
            glove_vectors = glove.vectors
            glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
            word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
            for word in word_dict:
                index = word_dict[word]
                if index != 0:
                    if word in glove_stoi:
                        word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
                    else:
                        random_vector = torch.zeros(config.word_embedding_dim)
                        random_vector.normal_(mean=0, std=0.1)
                        word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            with open(word_embedding_file, 'wb') as word_embedding_f:
                pickle.dump(word_embedding_vectors, word_embedding_f)

    def __init__(self, config: Config):
        # preprocess data
        MIND_Corpus.preprocess(config)
        with open('user_ID.json', 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('news_ID.json', 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('category.json', 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('subCategory.json', 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(
                config.max_title_length) + '-' + str(config.max_body_length) + '.json', 'r',
                  encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(
                config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(
            config.max_body_length) + '.pkl', 'rb') as word_embedding_f:
            self.word_emb = pickle.load(word_embedding_f)

        # meta data
        self.dataset = config.dataset
        self.news_encoder = config.news_encoder
        self.top = config.top
        self.negative_sample_num = config.negative_sample_num  # negative sample number for training
        self.max_history_num = config.max_history_num  # max history number for each training user
        self.max_title_length = config.max_title_length  # max title length for each news text
        self.max_body_length = config.max_body_length  # max body length for each news text
        self.news_category = np.zeros([self.news_num], dtype=np.int32)  # [news_num]
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)  # [news_num]
        self.news_title_text = np.zeros([self.news_num, self.max_title_length],
                                        dtype=np.int32)  # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length],
                                        dtype=np.float32)  # [news_num, max_title_length]
        self.news_body_text = np.zeros([self.news_num, self.max_body_length],
                                       dtype=np.int32)  # [news_num, max_body_length]
        self.news_body_mask = np.zeros([self.news_num, self.max_body_length],
                                       dtype=np.float32)  # [news_num, max_body_length]
        self.news_mask = np.zeros([self.news_num, self.max_title_length, self.max_title_length + self.max_body_length],
                                  dtype=np.float32)  # [news_num, max_title_length, max_title_length + max_body_length]: To implement selection module efficiently, we use mask
        self.train_behaviors = []  # [user_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []  # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []  # index for dev
        self.test_behaviors = []  # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []
        self.title_word_num = 0
        self.body_word_num = 0

        # generate news meta data
        news_ID_set = set(['<PAD>'])
        news_lines = []
        for root in [config.train_root, config.dev_root, config.test_root]:
            with open(os.path.join(root, 'news_with_body.tsv'), 'r', encoding='utf-8') as train_news_f:
                for line in train_news_f:
                    news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities, body = line.split(
                        '\t')
                    if news_ID not in news_ID_set:
                        news_lines.append(line)
                        news_ID_set.add(news_ID)

        assert self.news_num == len(news_ID_set), 'news num mismatch %d v.s. %d' % (self.news_num, len(news_ID_set))
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities, body = line.split(
                '\t')
            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict[category] if category in self.category_dict else 0
            self.news_subCategory[index] = self.subCategory_dict[
                subCategory] if subCategory in self.subCategory_dict else 0
            # words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            word_id = tokenizer(title.lower(), return_tensors="np", max_length=512).input_ids
            words = tokenizer.convert_ids_to_tokens(word_id.squeeze(), skip_special_tokens=True)

            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_title_text[index][i] = self.word_dict[word]
                else:
                    self.news_title_text[index][i] = 1
                self.news_title_mask[index][i] = 1
            self.title_word_num += len(words)

            # words = pat.findall(body.lower()) if config.tokenizer == 'MIND' else word_tokenize(body.lower())
            word_id = tokenizer(body.lower(), return_tensors="np", max_length=512).input_ids
            words = tokenizer.convert_ids_to_tokens(word_id.squeeze(), skip_special_tokens=True)

            for i, word in enumerate(words):
                if i == self.max_body_length:
                    break
                if is_number(word):
                    self.news_body_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_body_text[index][i] = self.word_dict[word]
                else:
                    self.news_body_text[index][i] = 1
                self.news_body_mask[index][i] = 1
        self.news_title_mask[0][0] = 1  # for <PAD> news
        self.news_body_mask[0][0] = 1  # for <PAD> news

        # cosine similarity
        def sim_matrix(a, b, eps=1e-8):
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.clamp(a_n, min=eps)
            b_norm = b / torch.clamp(b_n, min=eps)
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
            return sim_mt

        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size,
                                           embedding_dim=config.word_embedding_dim).cuda()
        self.word_embedding.weight.data.copy_(self.word_emb)
        for index, (t, b) in enumerate(zip(self.news_title_text[1:], self.news_body_text[1:])):
            # title word self edge
            mat = np.array([[0] * (self.max_body_length + i) + [1] + [0] * (self.max_title_length - i - 1) for i, j in
                            enumerate(t)])
            t = torch.tensor(t).cuda(non_blocking=True)
            b = torch.tensor(b).cuda(non_blocking=True)
            t = t.long()
            b = b.long()
            emb_t = self.word_embedding(t)
            emb_b = self.word_embedding(b)

            sim_mat = sim_matrix(emb_t, emb_b)

            t = t.cpu()
            b = b.cpu()
            not_t_pad = np.where(t != 0)[0]
            not_b_pad = np.where(b != 0)[0]
            if not_b_pad.any():
                link_b = np.argsort(
                    sim_mat[not_t_pad[0]:not_t_pad[-1] + 1, :self.max_body_length].cpu().detach().numpy(), axis=-1)[:,
                         ::-1][:, :self.top]
                for i, j in enumerate(link_b):
                    b_list = np.intersect1d(not_b_pad, j)
                    if b_list.any():
                        mat[i, b_list] = 1

            self.news_mask[index + 1] = mat

        # generate behavior meta data
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
            for behavior_index, line in enumerate(train_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(self.news_ID_dict[impression[:-2]])
                    else:
                        non_click_impressions.append(self.news_ID_dict[impression[:-2]])
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_position = np.array(list(range(50 - padding_num + 1, 1, -1)) + [0] * padding_num,
                                                     dtype=np.float32)
                    user_history_mask = np.zeros([self.max_history_num], dtype=np.float32)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1.0

                    for click_impression in click_impressions:
                        self.train_behaviors.append(
                            [self.user_ID_dict[user_ID], user_history, user_history_mask, click_impression,
                             non_click_impressions, behavior_index, user_history_position])
                else:
                    for click_impression in click_impressions:
                        self.train_behaviors.append(
                            [self.user_ID_dict[user_ID], [0 for _ in range(self.max_history_num)],
                             np.zeros([self.max_history_num], dtype=np.float32), click_impression,
                             non_click_impressions, behavior_index, np.zeros([self.max_history_num], dtype=np.float32),
                             np.zeros([config.subCategory_num, self.max_history_num], dtype=np.float32)])

        with open(os.path.join(config.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_behaviors_f:
            for dev_ID, line in enumerate(dev_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_position = np.array(list(range(50 - padding_num + 1, 1, -1)) + [0] * padding_num,
                                                     dtype=np.float32)
                    user_history_mask = np.zeros([self.max_history_num], dtype=np.float32)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1.0

                    self.dev_indices.append(dev_ID)
                    self.dev_behaviors.append(
                        [self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history,
                         user_history_mask,
                         [self.news_ID_dict[impression[:-2]] for impression in impressions.strip().split(' ')], dev_ID,
                         user_history_position])
                else:
                    self.dev_indices.append(dev_ID)
                    self.dev_behaviors.append(
                        [self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history,
                         user_history_mask,
                         [self.news_ID_dict[impression[:-2]] for impression in impressions.strip().split(' ')], dev_ID,
                         np.zeros([self.max_history_num], dtype=np.float32)])

        with open(os.path.join(config.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_behaviors_f:
            for test_ID, line in enumerate(test_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_position = np.array(list(range(50 - padding_num + 1, 1, -1)) + [0] * padding_num,
                                                     dtype=np.float32)
                    user_history_mask = np.zeros([self.max_history_num], dtype=np.float32)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1.0
                    self.test_indices.append(test_ID)
                    if self.dataset != 'large':
                        self.test_behaviors.append(
                            [self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history,
                             user_history_mask,
                             [self.news_ID_dict[impression[:-2]] for impression in impressions.strip().split(' ')],
                             test_ID, user_history_position])
                    else:
                        self.test_behaviors.append(
                            [self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history,
                             user_history_mask,
                             [self.news_ID_dict[impression] for impression in impressions.strip().split(' ')], test_ID,
                             user_history_position])
                else:
                    self.test_indices.append(test_ID)
                    if self.dataset != 'large':
                        self.test_behaviors.append(
                            [self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history,
                             user_history_mask,
                             [self.news_ID_dict[impression[:-2]] for impression in impressions.strip().split(' ')],
                             test_ID, np.zeros([self.max_history_num], dtype=np.float32)])
                    else:
                        self.test_behaviors.append(
                            [self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history,
                             user_history_mask,
                             [self.news_ID_dict[impression] for impression in impressions.strip().split(' ')], test_ID,
                             np.zeros([self.max_history_num], dtype=np.float32)])
