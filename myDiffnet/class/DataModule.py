'''
    author: Jiyao WANG
    e-mail: jiyaowang130@gmail.com
    release date:
'''

from collections import defaultdict
import numpy as np
from time import time
import random
import pickle as pickle


class DataModule():

    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0


    ###########################################  Initalize Procedures ############################################
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrix()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            self.readSocialNeighbors()
            self.generateSocialNeighborsSparseMatrix()
            data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list
        return data_dict


    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generatingTrainNegative()


    def initiateRankingVal(self):
        self.readData()
        self.arrangePositiveData()
        self.readValNegative()


    def initiateRankingEva(self):
        self.readData()
        self.arrangePositiveData()
        self.readEvaNegative()


    ###########################################  Ranking ############################################
    def readData(self):
        f = open(self.filename)
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split('\t')
            hash_data[(int(arr[0]), int(arr[1]))] = 1  # implicit feedback
            total_user_list.add(int(arr[0]))
        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data


    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)
            self.positive_data = positive_data
            self.total_data = len(total_data)


    '''
            This function is designed for the train/val/test negative generating/reading section
    '''


    def generatingTrainNegative(self):
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for u, i in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        self.negative_data = negative_data
        self.terminal_flag = 1


    def readValNegative(self):
        total_user_list = self.total_user_list
        eva_negative_data = dict()

        with open(self.conf.data_dir + '/Neg_val.pkl', 'rb') as f:
            val_set = pickle.load(f)
        val_set = dict(val_set)
        for u in total_user_list:
            eva_negative_data[u] = val_set[u]
        self.eva_negative_data = eva_negative_data


    def readEvaNegative(self):
        total_user_list = self.total_user_list
        eva_negative_data = dict()

        with open(self.conf.data_dir + '/Neg_test.pkl', 'rb') as f:
            test_set = pickle.load(f)
        test_set = dict(test_set)
        for u in total_user_list:
            eva_negative_data[u] = test_set[u]
        self.eva_negative_data = eva_negative_data


    '''
        This function designes for the val/test section, compute loss
    '''


    def getValRankingOneBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])


    def getTrainRankingBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index
        batch_size = self.conf.training_batch_size

        user_list, item_list, labels_list = [], [], []

        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index + batch_size]
            self.index = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0
            self.terminal_flag = 0

        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])


    def getEvaPositiveBatch(self):
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_index_dict = index_dict


    def getEvaRankingBatch(self):
        batch_size = self.conf.evaluate_batch_size
        num_evaluate = self.conf.num_evaluate
        eva_negative_data = self.eva_negative_data
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index + batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u] * num_evaluate)
            item_list.extend(eva_negative_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag


    def generateSocialNeighborsSparseMatrix(self):
        social_neighbors = self.social_neighbors
        social_neighbors_indices_list = []
        social_neighbors_value_list = []
        social_neighbors_dict = defaultdict(list)
        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u])

        user_list = sorted(list(social_neighbors.keys()))
        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_value_list.append((1.0 / len(social_neighbors_dict[user])))
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)
        self.social_neighbors_values_list = np.array(social_neighbors_value_list).astype(np.float32)


    def generateConsumedItemsSparseMatrix(self):
        positive_data = self.positive_data
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_dict = defaultdict(list)
        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])
        user_list = sorted(list(positive_data.keys()))
        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0 / len(consumed_items_dict[u]))
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64)
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)
