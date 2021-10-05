import numpy as np
import tensorflow
import time
import math
import os
from collections import defaultdict


tf = tensorflow.compat.v1


class AliasTable:
    def __init__(self, nums):
        self.n = len(nums)
        self.alias = list(range(self.n))
        sum_nums = sum(nums)
        self.prob = [i*self.n/sum_nums for i in nums]
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

    def sampling(self, times=1):
        def one_time():
            pos = np.random.randint(self.n)
            return pos if np.random.rand() < self.prob[pos] else self.alias[pos]
        return one_time() if times == 1 else [one_time() for _ in range(times)]


class Model:
    def __init__(self, data_fold, save_fold, session):
        init_time = time.time()
        self.first_data, self.second_data = os.listdir(data_fold)
        self.first_data = os.path.join(data_fold, self.first_data)
        self.second_data = os.path.join(data_fold, self.second_data)
        self.save_fold = save_fold
        self.session = session

        self.embedding_size = 128
        self.learning_rate = 0.5
        self.num_batches = 10**5
        self.batch_size = 1024
        self.loss_stage = max(1, self.num_batches // 10000)
        self.exact_stage = max(1, self.num_batches // 1000)
        self.negative = 5
        self.web_gap = 0.1
        self.hits = 1

        self.first_info = []
        self.second_info = []
        self.first_name = []
        self.second_name = []
        self.get_info()

        self.first_count = len(self.first_info)
        self.second_count = len(self.second_info)

        self.web_index = {}
        self.web_count = 0
        self.web_first = []
        self.web_second = []
        self.first_edge = []
        self.first_edge_sample = AliasTable([1])
        self.second_edge = []
        self.second_edge_sample = AliasTable([1])
        self.make_graph()

        init_width = 0.5 / self.embedding_size
        first_emb = tf.random_uniform(shape=(self.first_count, self.embedding_size),
                                      minval=-init_width, maxval=init_width, dtype=tf.float32)
        self.first_emb = tf.Variable(first_emb, name='first_emb')
        second_emb = tf.random_uniform(shape=(self.second_count, self.embedding_size),
                                       minval=-init_width, maxval=init_width, dtype=tf.float32)
        self.second_emb = tf.Variable(second_emb, name='second_emb')

        self.start_ph = tf.placeholder(shape=(self.batch_size*self.negative), dtype=tf.int32, name='start_ph')
        self.end_ph = tf.placeholder(shape=(self.batch_size*self.negative), dtype=tf.int32, name='end_ph')
        self.weight_ph = tf.placeholder(shape=(self.batch_size*self.negative), dtype=tf.float32,
                                        name='weight_ph')

        self.loss = self.make_loss()
        self.optimizer = self.make_optimizer()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        print('initial time: %s, first count: %s, second count: %s'
              % (time.time() - init_time, self.first_count, self.second_count))

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
        self.web_first = [{-1} for _ in range(self.web_count)]
        self.web_second = [{-1}for _ in range(self.web_count)]
        for i in range(self.first_count):
            for j in first_web_tf[i]:
                web_idf[j] += 1
                self.web_first[j].add(i)
        for i in range(self.second_count):
            for j in second_web_tf[i]:
                web_idf[j] += 1
                self.web_second[j].add(i)
        [i.remove(-1)for i in self.web_first], [i.remove(-1)for i in self.web_second]
        sum_idf = sum(web_idf)
        web_idf = [math.log(sum_idf / web_idf[i])for i in range(self.web_count)]

        first_edge_weight = []
        second_edge_weight = []
        for i in range(self.first_count):
            for j in first_web_tf[i]:
                self.first_edge.append((i, j))
                first_edge_weight.append(first_web_tf[i][j]*web_idf[j])
        for i in range(self.second_count):
            for j in second_web_tf[i]:
                self.second_edge.append((i, j))
                second_edge_weight.append(second_web_tf[i][j]*web_idf[j])

        def soft_max(xx):
            return self.session.run(tf.nn.softmax(np.array(xx)))

        self.first_edge_sample = AliasTable(soft_max(first_edge_weight))
        self.second_edge_sample = AliasTable(soft_max(second_edge_weight))

    def make_loss(self):
        start_emb = tf.nn.embedding_lookup(self.first_emb, self.start_ph)
        end_emb = tf.nn.embedding_lookup(self.second_emb, self.end_ph)
        inner_product = tf.reduce_sum(tf.multiply(start_emb, end_emb), axis=1)
        #return -tf.reduce_mean(tf.multiply(tf.log_sigmoid(inner_product), self.weight_ph))
        return -tf.reduce_mean(tf.log_sigmoid(self.weight_ph * inner_product))
    def make_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fetch_batch(self):
        start = []
        end = []
        weight = []
        first_choice = range(self.first_count)
        second_choice = range(self.second_count)
        for i in self.first_edge_sample.sampling(self.batch_size//2):
            first, web = self.first_edge[i]
            for j in range(self.negative):
                second = np.random.choice(second_choice)
                start.append(first)
                end.append(second)
                if second in self.web_second[web]:
                    weight.append(1)
                else:
                    weight.append(-1)
        for i in self.second_edge_sample.sampling(self.batch_size//2):
            second, web = self.second_edge[i]
            for j in range(self.negative):
                first = np.random.choice(first_choice)
                start.append(first)
                end.append(second)
                if first in self.web_first[web]:
                    weight.append(1)
                else:
                    weight.append(-1)
        return start, end, weight

    def run_train(self):
        self.save_init()
        total_sample_time, total_train_time = 0, 0
        print('start training')

        for i in range(1, self.num_batches+1):
            sample_time = time.time()
            start, end, weight = self.fetch_batch()
            total_sample_time += time.time()-sample_time
            feed_dict = {self.start_ph: start, self.end_ph: end, self.weight_ph: weight}
            train_time = time.time()
            self.session.run(self.optimizer, feed_dict=feed_dict)
            self.learning_rate = max(0.0001, self.learning_rate*(1-i/self.num_batches))
            total_train_time += time.time()-train_time
            if not i % self.loss_stage:
                loss = self.session.run(self.loss, feed_dict=feed_dict)
                print('batch: %s, sampling_time:%.2f, train_time:%.2f, loss:%.4f'
                      % (i, total_sample_time, total_train_time, loss))
            if not i % self.exact_stage:
                self.cal_exact_rate()

        print('finish training')
        self.save_res()

    def save_init(self):
        pass

    def save_res(self):
        pass

    def cal_exact_rate(self):
        def make_similarity_matrix():
            a = []
            b = []
            for p in range(self.first_count):
                for q in range(self.second_count):
                    a.append(p)
                    b.append(q)
            a_emb = tf.nn.embedding_lookup(self.first_emb, a)
            b_emb = tf.nn.embedding_lookup(self.second_emb, b)
            sim_val = self.session.run(tf.sigmoid(tf.reduce_sum(tf.multiply(a_emb, b_emb), axis=1)))
            # print(sim_val)
            return sim_val.reshape((self.first_count, self.second_count))

        sim_time = time.time()
        print('similarity calculation start')
        similarity_matrix = make_similarity_matrix()
        print('similarity matrix time: %.2f' % (time.time() - sim_time))

        sim_time = time.time()
        first_exact_rate = 0
        for i in range(self.first_count):
            order = sorted(range(self.second_count), key=lambda x: -similarity_matrix[i][x])
            for j in range(self.hits):
                if self.first_name[i] == self.second_name[order[j]]:
                    first_exact_rate += 1
                    break
        second_exact_rate = 0
        for i in range(self.second_count):
            order = sorted(range(self.first_count), key=lambda x: -similarity_matrix[x][i])
            for j in range(self.hits):
                if self.second_name[i] == self.first_name[order[j]]:
                    second_exact_rate += 1
                    break
        print('exact time: %.2f' % (time.time() - sim_time))

        first_exact_rate /= self.first_count
        second_exact_rate /= self.second_count
        print('first exact rate: %.4f' % first_exact_rate)
        print('second exact rate: %.4f' % second_exact_rate)


def main(data_fold, save_fold):
    total_time = time.time()
    with tf.Graph().as_default(), tf.Session() as sess:
        ob = Model(data_fold, save_fold, sess)
        ob.run_train()
    print('total time: %.2f' % (time.time()-total_time))


if __name__ == '__main__':
    main(data_fold='Data2', save_fold='Result')
