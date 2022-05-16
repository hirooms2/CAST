import os
import argparse
import time
import torch
import random
import numpy as np
import json


class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='Neural news recommendation')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='CAST', help='News encoder')
        parser.add_argument('--user_encoder', type=str, default='CAST', help='User encoder')
        parser.add_argument('--dev_model_path', type=str, default='', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Specific test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=-1, help='Seed for random number generator')
        parser.add_argument('--config_file', type=str, default='', help='Config file path')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='MIND_small', choices=['MIND_small', 'MIND_large'],
                            help='Dataset type')

        parser.add_argument('--train_root', type=str, default='MIND/train', help='Directory root of training data')
        parser.add_argument('--dev_root', type=str, default='MIND/dev', help='Directory root of dev data')
        parser.add_argument('--test_root', type=str, default='MIND/test', help='Directory root of test data')

        parser.add_argument('--tokenizer', type=str, default='BERT', choices=['MIND', 'NLTK', 'BERT'],
                            help='Sentence tokenizer')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=30, help='Sentence truncate length for title')
        parser.add_argument('--max_body_length', type=int, default=100, help='Sentence truncate length for body')
        parser.add_argument('--top', type=int, default=250, help='degree of Sparse')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4,
                            help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50,
                            help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=5, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4,
                            help='Gradient clip norm (non-positive value for no clipping)')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='auc', choices=['auc', 'mrr', 'ndcg', 'ndcg10'],
                            help='Validation criterion to select model')
        # Model config
        parser.add_argument('--transform_dim', type=int, default=400, help='Transformation dimension of user encoder')
        parser.add_argument('--position_dim', type=int, default=300, help='Positional dimension of user encoder')
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300],
                            help='Word embedding dimension')
        parser.add_argument('--attention_dim', type=int, default=200, help="Attention dimension")
        parser.add_argument('--head_num', type=int, default=20, help='Head number of multi-head self-attention')
        parser.add_argument('--head_dim', type=int, default=20, help='Head dimension of multi-head self-attention')
        parser.add_argument('--category_embedding_dim', type=int, default=50, help='Category embedding dimension')
        parser.add_argument('--subCategory_embedding_dim', type=int, default=50, help='SubCategory embedding dimension')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')

        self.attribute_dict = dict(vars(parser.parse_args()))
        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file: ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    for attribute in self.attribute_dict:
                        if attribute in configs:
                            setattr(self, attribute, configs[attribute])
            else:
                raise Exception('Config file does not exist: ' + self.config_file)
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute in self.attribute_dict:
            print(attribute + ' : ' + str(getattr(self, attribute)))
        print('*' * 32 + ' Experiment setting ' + '*' * 32)

        self.train_root = 'datasets/%s/train/' % self.attribute_dict['dataset']
        self.test_root = 'datasets/%s/test/' % self.attribute_dict['dataset']
        self.dev_root = 'datasets/%s/dev/' % self.attribute_dict['dataset']

    def set_cuda(self):
        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # For reproducibility

    def preliminary_setup(self):

        print(self.train_root)
        print(self.test_root)

        required_dataset_files = [
            os.path.join(self.train_root, 'behaviors.tsv'), os.path.join(self.train_root, 'news.tsv'),
            os.path.join(self.dev_root, 'behaviors.tsv'), os.path.join(self.dev_root, 'news.tsv'),
            os.path.join(self.test_root, 'behaviors.tsv'), os.path.join(self.test_root, 'news.tsv')
        ]

        print(required_dataset_files)

        assert all([os.path.exists(f) for f in required_dataset_files]), "Download the dataset"

        model_name = self.news_encoder + '-' + self.user_encoder
        mkdirs = lambda p: os.makedirs(p) if not os.path.exists(p) else None
        mkdirs('./configs/' + model_name)
        mkdirs('./models/' + model_name)
        mkdirs('./best_model/' + model_name)
        mkdirs('./dev/ref')
        mkdirs('./dev/res/' + model_name)
        mkdirs('./test/ref')
        mkdirs('./test/res/' + model_name)
        mkdirs('./results/' + model_name)
        if not os.path.exists('./dev/ref/truth.txt'):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('./dev/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(
                            ('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if not os.path.exists('./test/ref/truth.txt'):
            with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                with open('./test/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for test_ID, line in enumerate(test_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(
                            ('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))

    def __init__(self):
        self.parse_argument()
        self.set_cuda()
        self.preliminary_setup()


if __name__ == '__main__':
    config = Config()
