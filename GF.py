import numpy as np
import tensorflow
import time
import math
import os
from collections import defaultdict


tf = tensorflow.compat.v1
t = time.time()

class Model:
    def __init__(self, data_fold, save_fold, session):
        init_time = time.time()
        self.first_data, self.second_data = os.listdir(data_fold)
        self.first_data = os.path.join(data_fold, self.first_data)
        self.second_data = os.path.join(data_fold, self.second_data)
        self.save_fold = save_fold
        self.session = session

        self.embedding_size = 64
        self.learning_rate = 0.001
        self.num_batches = 10**6
        self.batch_size = 128
        self.loss_stage = 10
        self.exact_stage = 100
        self.web_gap = 0.1
        self.hits = 10
        self.a = 0.01

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
        self.matrix = []
        self.make_graph()

        init_width = 0.5 / self.embedding_size
        all_emb = tf.random_uniform(shape=(self.total_count, self.embedding_size),
                                    minval=-init_width, maxval=init_width, dtype=tf.float32)
        self.all_emb = tf.Variable(all_emb, name='all_emb')

        ph_size = self.batch_size*self.total_count
        self.node_ph = tf.placeholder(shape=self.batch_size, dtype=tf.int32, name='node_ph')
        self.start_ph = tf.placeholder(shape=ph_size, dtype=tf.int32, name='start_ph')
        self.end_ph = tf.placeholder(shape=ph_size, dtype=tf.int32, name='end_ph')
        self.weight_ph = tf.placeholder(shape=ph_size, dtype=tf.float32, name='weight_ph')

        self.loss = self.make_loss()
        self.optimizer = self.make_optimizer()
        self.init_time = time.time() - init_time
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
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

        self.matrix = [[0.0]*self.total_count for _ in range(self.total_count)]
        for i in range(self.first_count):
            for j in first_web_tf[i]:
                self.matrix[i][j+self.user_count] = first_web_tf[i][j]*web_idf[j]
                self.matrix[j+self.user_count][i] = first_web_tf[i][j]*web_idf[j]
        for i in range(self.second_count):
            for j in second_web_tf[i]:
                self.matrix[i+self.first_count][j+self.user_count] = second_web_tf[i][j]*web_idf[j]
                self.matrix[j+self.user_count][i+self.first_count] = second_web_tf[i][j]*web_idf[j]

    def make_loss(self):
        start_emb = tf.nn.embedding_lookup(self.all_emb, self.start_ph)
        end_emb = tf.nn.embedding_lookup(self.all_emb, self.end_ph)
        inner_product = tf.reduce_sum(tf.multiply(start_emb, end_emb), axis=1)
        ro = tf.subtract(self.weight_ph, inner_product)
        ro = tf.reduce_sum(tf.multiply(ro, ro))
        jo = tf.nn.embedding_lookup(self.all_emb, self.node_ph)
        jo = tf.reduce_sum(tf.multiply(jo, jo))
        return ro+self.a*jo

    def make_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fetch_batch(self):
        choices = list(range(self.total_count))
        node = np.random.choice(choices, size=self.batch_size)
        start = []
        end = []
        weight = []
        for i in node:
            start.extend([i]*self.total_count)
            end.extend(range(self.total_count))
            weight.extend(self.matrix[i])
        return node, start, end, weight

    def run_train(self):
        self.save_init()
        total_sample_time, total_train_time = 0, 0
        print('start training')

        for i in range(1, self.num_batches+1):
            sample_time = time.time()
            node, start, end, weight = self.fetch_batch()
            total_sample_time += time.time()-sample_time
            feed_dict = {self.node_ph: node, self.start_ph: start, self.end_ph: end, self.weight_ph: weight}
            train_time = time.time()
            self.session.run(self.optimizer, feed_dict=feed_dict)
            total_train_time += time.time()-train_time
            self.learning_rate = max(0.0001, self.learning_rate*(1-i/self.num_batches))
            if not i % self.loss_stage:
                loss = self.session.run(self.loss, feed_dict=feed_dict)
                print('batch: %s, sampling_time:%.2f, train_time:%.2f, loss:%10f'
                      % (i, total_sample_time, total_train_time, loss))
            if not i % self.exact_stage:
                print('batch: %s, sampling_time:%.2f, train_time:%.2f, running time: %.2f'
                      % (i, total_sample_time, total_train_time,total_sample_time+total_train_time+self.init_time))
                self.cal_exact_rate()

        print('finish training')
        self.save_res()

    def save_init(self):
        pass

    def save_res(self):
        pass

    def cal_exact_rate(self):
        def make_similarity_matrix():
            aa = []
            bb = []
            for p in range(self.first_count):
                aa.extend([p]*self.second_count)
                bb.extend(range(self.second_count, self.user_count))
            a_emb = tf.nn.embedding_lookup(self.all_emb, aa)
            b_emb = tf.nn.embedding_lookup(self.all_emb, bb)
            # sim_val = self.session.run(tf.sigmoid(tf.reduce_sum(tf.multiply(a_emb, b_emb), axis=1)))
            sim_val = self.session.run(tf.reduce_sum(tf.multiply(a_emb, b_emb), axis=1))
            # print(sim_val)
            return sim_val.reshape((self.first_count, self.second_count))

        sim_time = time.time()
        # print('similarity calculation start')
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
        print('first / second exact rate: %.4f / %.4f' % (first_exact_rate, second_exact_rate))
        print('whole_time: %2f ' % (time.time() - t))

def main(data_fold, save_fold):
    total_time = time.time()
    with tf.Graph().as_default(), tf.Session() as sess:
        ob = Model(data_fold, save_fold, sess)
        ob.run_train()
    print('total time: %.2f' % (time.time()-total_time))


if __name__ == '__main__':
    main(data_fold='Data2', save_fold='Result')


