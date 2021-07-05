'''
    author: Jiyao WANG
    e-mail: jiyaowang130@gmail.com
    release date:
'''

import math
import numpy as np


class Evaluate():

    def __init__(self, conf):
        self.conf = conf


    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg += math.log(2) / math.log(i + 2)
        return idcg


    def getDcg(self, value):
        dcg = math.log(2) / math.log(value + 2)
        return dcg


    def getHr(self, value):
        hit = 1.0
        return hit


    def evaluateRankingPerformance(self, evaluate_index_dict, evaluate_real_rating_matrix,
                                   evaluate_predict_rating_matrix, topk, num_procs, exp_flag=0, sp_name=None,
                                   result_file=None):
        user_list = list(evaluate_index_dict.keys())
        batch_size = len(user_list) / num_procs

        hr_list, ndcg_list = [], []
        index = 0
        for _ in range(num_procs):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index + batch_size]
                index += batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            tmp_hr_list, tmp_ndcg_list = self.getHrNdcgProc(evaluate_index_dict, evaluate_real_rating_matrix,
                                                            evaluate_predict_rating_matrix, topk, batch_user_list)
            hr_list.extend(tmp_hr_list)
            ndcg_list.extend(tmp_ndcg_list)
        return np.mean(hr_list), np.mean(ndcg_list)


    def getHrNdcgProc(self, evaluate_index_dict, evaluate_real_rating_matrix,
                      evaluate_predict_rating_matrix, topk, user_list):
        tmp_hr_list, tmp_ndcg_list = [], []
        for u in user_list:
            real_item_index_list = evaluate_index_dict[u]
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
            positive_length = len(real_item_rating_list)
            target_length = min(positive_length, topk)

            predict_rating_list = evaluate_predict_rating_matrix[u]
            real_item_rating_list.extend(predict_rating_list)
            sort_index = np.argsort(real_item_rating_list)
            sort_index = sort_index[::-1]

            user_hr_list = []
            user_ndcg_list = []
            hits_num = 0
            for idx in range(topk):
                ranking = sort_index[idx]
                if ranking < positive_length:
                    hits_num += 1
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))

            idcg = self.getIdcg(target_length)
            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            tmp_hr_list.append(tmp_hr)
            tmp_ndcg_list.append((tmp_ndcg))

            return tmp_hr_list, tmp_ndcg_list
