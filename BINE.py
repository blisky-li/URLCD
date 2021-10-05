import numpy as np
import time
import math
import os
import heapq
from collections import defaultdict
from gensim.models import Word2Vec

t = time.time()

class AliasTable:
    def __init__(self, prob, obj):
        self.n = len(prob)
        if len(obj) == self.n:
            self.obj = obj
        else:
            self.obj = range(self.n)
        sum_prob = sum(prob)
        if not sum_prob:
            prob = [1] * self.n
            sum_prob = self.n
        self.alias = list(range(self.n))
        self.prob = [i * self.n / sum_prob for i in prob]
        small, large = [], []
        for i in range(self.n):
            if self.prob[i] > 1:
                large.append(i)
            else:
                small.append(i)
        while small and large:
            sm, la = small.pop(), large.pop()
            self.alias[sm] = la
            self.prob[la] -= 1-self.prob[sm]
            if self.prob[la] > 1:
                large.append(la)
            else:
                small.append(la)

    def sample_one(self):
        pos = np.random.randint(self.n)
        ap = pos if np.random.rand() < self.prob[pos] else self.alias[pos]
        return self.obj[ap]

    def sample_n(self, n_times):
        return [self.sample_one() for _ in range(n_times)]


class Model:
    def __init__(self, data_fold, save_fold):
        init_time = time.time()
        self.first_data, self.second_data = os.listdir(data_fold)
        self.first_data = os.path.join(data_fold, self.first_data)
        self.second_data = os.path.join(data_fold, self.second_data)
        self.save_fold = save_fold

        self.embedding_size = 64
        self.num_batches = 100000
        self.batch_size = 512
        self.exact_stage = 800
        self.web_gap = 0.1
        self.hits = 5
        self.alpha = 0.01
        self.percentage = 0.15

        self.first_info = []
        self.second_info = []
        self.first_name = []
        self.second_name = []
        self.get_info()

        self.first_count = len(self.first_info)
        self.second_count = len(self.second_info)
        self.user_count = self.first_count+self.second_count
        self.total_count = 0

        self.web_index = {}
        self.web_count = 0
        self.all_edge = []
        self.make_graph()

        self.neighbor_sample = []
        self.node_sample_user = AliasTable([1], [1])
        self.node_sample_web = AliasTable([1], [1])
        self.make_sample()

        self.model = Word2Vec([[str(i)]for i in range(self.total_count)],
                              vector_size=self.embedding_size, window=3, min_count=1, sg=1)  # sg:1为skip-gram，2为CBOW
        self.init_time = time.time() - init_time
        print('initial time: %s, first count: %s, second count: %s, web count: %s'
              % (time.time() - init_time, self.first_count, self.second_count, self.web_count))

    def get_info(self):
        for i in os.listdir(self.first_data):
            with open(os.path.join(self.first_data, i), 'r') as f:
                cur_info = []
                for j in f.readlines():
                    piece = j.split(',')
                    if len(piece) == 4:
                        name, x, y, t = piece
                        cur_info.append((name, float(x), float(y), int(t)))
                if cur_info:
                    self.first_info.append(cur_info)
                    self.first_name.append(cur_info[0][0])
        for i in os.listdir(self.second_data):
            with open(os.path.join(self.second_data, i), 'r') as f:
                cur_info = []
                for j in f.readlines():
                    piece = j.split(',')
                    if len(piece) == 4:
                        name, x, y, t = piece
                        cur_info.append((name, float(x), float(y), int(t)))
                if cur_info:
                    self.second_info.append(cur_info)
                    self.second_name.append(cur_info[0][0])

    def make_graph(self):
        def to_cell(xx, yy):
            return '%s-%s' % (int(xx / self.web_gap), int(yy / self.web_gap))

        first_web_tf = []
        second_web_tf = []
        for i in self.first_info:
            cur_info = defaultdict(float)
            for name, x, y, t in i:
                cur_cell = to_cell(x, y)
                if cur_cell not in self.web_index:
                    self.web_index[cur_cell] = self.web_count
                    self.web_count += 1
                cur_info[self.web_index[cur_cell]] += 1
            cur_sum = sum(cur_info.values())
            for j in cur_info:
                cur_info[j] /= cur_sum
            first_web_tf.append(cur_info)
        for i in self.second_info:
            cur_info = defaultdict(float)
            for name, x, y, t in i:
                cur_cell = to_cell(x, y)
                if cur_cell not in self.web_index:
                    self.web_index[cur_cell] = self.web_count
                    self.web_count += 1
                cur_info[self.web_index[cur_cell]] += 1
            cur_sum = sum(cur_info.values())
            for j in cur_info:
                cur_info[j] /= cur_sum
            second_web_tf.append(cur_info)

        web_idf = [0]*self.web_count
        self.total_count = self.user_count + self.web_count
        for i in range(self.first_count):
            for j in first_web_tf[i]:
                web_idf[j] += 1
        for i in range(self.second_count):
            for j in second_web_tf[i]:
                web_idf[j] += 1
        sum_idf = sum(web_idf)
        web_idf = [math.log(sum_idf / web_idf[i])for i in range(self.web_count)]

        for i in range(self.first_count):
            for j in first_web_tf[i]:
                self.all_edge.append((i, j+self.user_count, first_web_tf[i][j]*web_idf[j]))
                # self.all_edge.append((j+self.user_count, i, first_web_tf[i][j]*web_idf[j]))
        for i in range(self.second_count):
            for j in second_web_tf[i]:
                self.all_edge.append((i+self.first_count, j+self.user_count, second_web_tf[i][j]*web_idf[j]))
                # self.all_edge.append((j+self.user_count, i+self.first_count, second_web_tf[i][j]*web_idf[j]))

    def make_sample(self):
        neighbor = [[[], []]for _ in range(self.total_count)]
        degree = [0.0]*self.total_count
        for node_a, node_b, weight in self.all_edge:
            neighbor[node_a][0].append(node_b)
            neighbor[node_a][1].append(weight)
            neighbor[node_b][0].append(node_a)
            neighbor[node_b][1].append(weight)
            degree[node_a] += weight
            degree[node_b] += weight
        for i in range(self.total_count):
            self.neighbor_sample.append(AliasTable(neighbor[i][1], neighbor[i][0]))
        self.node_sample_user = AliasTable(degree[:self.user_count], list(range(self.user_count)))
        self.node_sample_web = AliasTable(degree[self.user_count:], list(range(self.user_count, self.total_count)))

    def fetch_path(self):
        paths = []
        for i in range(self.batch_size):
            cur = self.node_sample_user.sample_one()
            path = [cur]
            while np.random.random() > self.percentage:
                if np.random.random() > self.alpha:
                    cur = self.neighbor_sample[cur].sample_one()
                    cur = self.neighbor_sample[cur].sample_one()
                else:
                    cur = path[0]
                path.append(str(cur))
            paths.append(path)
        """
        for i in range(self.batch_size):
            cur = self.node_sample_web.sample_one()
            path = [cur]
            while np.random.random() > self.percentage:
                if np.random.random() > self.alpha:
                    cur = self.neighbor_sample[cur].sample_one()
                    cur = self.neighbor_sample[cur].sample_one()
                else:
                    cur = path[0]
                path.append(str(cur))
            paths.append(path)
        """
        return paths

    def run_train(self):
        self.save_init()
        total_sample_time, total_train_time = 0, 0
        print('start training')

        for i in range(1, self.num_batches+1):
            sample_time = time.time()
            paths = self.fetch_path()
            total_sample_time += time.time()-sample_time
            train_time = time.time()
            self.model.train(paths, total_words=self.total_count, epochs=1)
            total_train_time += time.time()-train_time
            if not i % self.exact_stage:
                print('batch: %s, sampling_time:%.2f, train_time:%.2f, running time: %.2f'
                      % (i, total_sample_time, total_train_time, total_sample_time+total_train_time+self.init_time))
                self.cal_exact_rate()

        print('finish training')
        self.save_res()

    def save_init(self):
        pass

    def save_res(self):
        pass

    def cal_exact_rate(self):
        sim_time = time.time()
        first_exact_rate = 0
        for i in range(self.first_count):
            order = [(-self.model.wv.similarity(str(i), str(x+self.first_count)), x)for x in range(self.second_count)]
            heapq.heapify(order)
            for j in range(self.hits):
                _, oj = heapq.heappop(order)
                if self.first_name[i] == self.second_name[oj]:
                    first_exact_rate += 1
        second_exact_rate = 0
        for i in range(self.second_count):
            order = [(-self.model.wv.similarity(str(i+self.first_count), str(x)), x)for x in range(self.first_count)]
            heapq.heapify(order)
            for j in range(self.hits):
                _, oj = heapq.heappop(order)
                if self.second_name[i] == self.first_name[oj]:
                    second_exact_rate += 1
        print('exact time: %.2f' % (time.time() - sim_time))

        first_exact_rate /= self.first_count
        second_exact_rate /= self.second_count
        print('first / second exact rate: %.4f / %.4f' % (first_exact_rate, second_exact_rate))
        print('exact rate: %.4f ' % ((first_exact_rate+second_exact_rate)/2))
        print('whole_time: %2f ' % (time.time() - t))

def main(data_fold, save_fold):
    total_time = time.time()
    ob = Model(data_fold, save_fold)
    ob.run_train()
    print('total time: %.2f' % (time.time()-total_time))


if __name__ == '__main__':
    main(data_fold='Data2', save_fold='Result')
