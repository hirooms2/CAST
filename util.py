import os
import platform
from config import Config
import torch
import torch.nn as nn
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring
from tqdm import tqdm


def compute_scores(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int, mode: str, result_file: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataset = MIND_DevTest_Dataset(mind_corpus, mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=batch_size // 8 if platform.system() == 'Linux' else 0, pin_memory=True)
    indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
    scores = []
    index = 0
    model.eval()
    with torch.no_grad():
        for (user_category, user_subCategory, user_title_text, user_title_mask, user_body_text, user_body_mask,
             user_news_mask, user_history_mask, user_history_position, \
             news_category, news_subCategory, news_title_text, news_title_mask, news_body_text, news_body_mask,
             news_mask) in tqdm(dataloader):
            user_category = user_category.cuda(non_blocking=True)  # [batch_size, max_history_num]
            user_subCategory = user_subCategory.cuda(non_blocking=True)  # [batch_size, max_history_num]
            user_title_text = user_title_text.cuda(non_blocking=True)  # [batch_size, max_history_num, max_title_length]
            user_title_mask = user_title_mask.cuda(non_blocking=True)  # [batch_size, max_history_num, max_title_length]
            user_body_text = user_body_text.cuda(non_blocking=True)  # [batch_size, max_history_num, max_body_length]
            user_body_mask = user_body_mask.cuda(non_blocking=True)  # [batch_size, max_history_num, max_body_length]
            user_news_mask = user_news_mask.cuda(
                non_blocking=True)  # [batch_size, max_history_num, max_title_length, max_title_length + max_body_length]
            user_history_mask = user_history_mask.cuda(non_blocking=True)  # [batch_size, max_history_num]
            user_history_position = user_history_position.cuda(non_blocking=True)  # [batch_size, max_history_num]
            news_category = news_category.cuda(non_blocking=True)  # [batch_size, 1 + negative_sample_num]
            news_subCategory = news_subCategory.cuda(non_blocking=True)  # [batch_size, 1 + negative_sample_num]
            news_title_text = news_title_text.cuda(
                non_blocking=True)  # [batch_size, 1 + negative_sample_num, max_title_length]
            news_title_mask = news_title_mask.cuda(
                non_blocking=True)  # [batch_size, 1 + negative_sample_num, max_title_length]
            news_body_text = news_body_text.cuda(
                non_blocking=True)  # [batch_size, 1 + negative_sample_num, max_body_length]
            news_body_mask = news_body_mask.cuda(
                non_blocking=True)  # [batch_size, 1 + negative_sample_num, max_body_length]
            news_mask = news_mask.cuda(
                non_blocking=True)  # [batch_size, 1 + negative_sample_num, max_title_length, max_title_length + max_body_length]
            batch_size = user_category.size(0)

            result = model(user_category, user_subCategory, user_title_text, user_title_mask, user_body_text,
                           user_body_mask, user_news_mask, user_history_mask, user_history_position, \
                           news_category, news_subCategory, news_title_text, news_title_mask, news_body_text,
                           news_body_mask, news_mask).squeeze(dim=1)  # [batch_size, 1 + negative_sample_num]
            result = result.view([-1, ]).cpu().tolist()
            scores.append(result)

    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for cnt, score in enumerate(scores):
        for e, val in enumerate(score):
            sub_scores[cnt].append([val, e])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))

    if mind_corpus.dataset == 'MIND_small':
        with open('./' + mode + '/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open(result_file, 'r',
                                                                                          encoding='utf-8') as result_f:
            auc, mrr, ndcg, ndcg10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg, ndcg10


def get_run_index(model: str):
    assert os.path.exists('./results/' + model), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir('./results/' + model):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open('./results/' + model + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1
