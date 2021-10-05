import numpy as np
# import math
import time
import math
import os
import tensorflow

from collections import defaultdict

l = []
tf = tensorflow.compat.v1


class Model:
    def __init__(self, data_fold, save_fold, session,em_size=128,lr=0.001):
        init_time = time.time()
        self.first_data, self.second_data = os.listdir(data_fold)
        self.first_data = os.path.join(data_fold, self.first_data)
        self.second_data = os.path.join(data_fold, self.second_data)
        self.save_fold = save_fold
        self.session = session

        self.embedding_size = em_size
        self.learning_rate = lr  # 0.1到0.005
        self.round = 40
        #self.stage = max(1, self.round // 10)
        self.stage = 10
        self.hits = 1

        self.first_info = []
        self.second_info = []
        self.first_name = []
        self.second_name = []
        self.get_info()

        self.first_count = len(self.first_info)
        self.second_count = len(self.second_info)

        self.init_first_second = [[0]*self.second_count for _ in range(self.first_count)]
        self.make_init_first_second(web_gap=1)
        self.make_init_first_second(web_gap=0.1)
        self.make_init_first_second(web_gap=0.01)
        self.make_init_first_second(web_gap=0.001)
        #self.make_init_first_second(web_gap=0.0001)
        """self.make_init_first_second(web_gap=0.5)
        self.make_init_first_second(web_gap=0.05)
        self.make_init_first_second(web_gap=0.005)"""
        # print(self.first_info)
        # print(first_web_tf)
        # print(first_name)
        # print(web_idf)
        # print(web_first)
        # print(web_second)
        # for i in first_second:
        #     print(i)
        # print()
        self.first_second = session.run(tf.nn.softmax(np.array(self.init_first_second)))

        init_width = 0.5 / self.embedding_size
        first_emb = tf.random_uniform(shape=(self.first_count, self.embedding_size),
                                      minval=-init_width, maxval=init_width,
                                      dtype=tf.float32, name='first_emb')
        self.first_emb = tf.Variable(first_emb)
        second_emb = tf.random_uniform(shape=(self.second_count, self.embedding_size),
                                       minval=-init_width, maxval=init_width,
                                       dtype=tf.float32, name='second_emb')
        self.second_emb = tf.Variable(second_emb)

        self.start_ph = tf.placeholder(shape=(self.first_count * self.second_count), dtype=tf.int32, name='start_ph')
        self.end_ph = tf.placeholder(shape=(self.first_count * self.second_count), dtype=tf.int32, name='end_ph')
        self.weight_ph = tf.placeholder(shape=(self.first_count * self.second_count), dtype=tf.float32,
                                        name='weight_ph')

        self.loss = self.make_loss()
        self.train_model = self.make_model()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        print('initial time: %s, first count: %s, second count: %s'
              % (time.time()-init_time, self.first_count, self.second_count))
        self.theory_exact_analyse()

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

    def make_init_first_second(self, web_gap):
        def to_cell(xx, yy):
            return '%s-%s' % (int(xx / web_gap), int(yy / web_gap))

        first_web_tf = []
        second_web_tf = []
        for i in self.first_info:
            cur_info = defaultdict(float)
            for j in i:
                cur_info[to_cell(j[1], j[2])] += 1
            cur_sum = sum(cur_info.values())
            for j in cur_info:
                cur_info[j] /= cur_sum
            first_web_tf.append(cur_info)
        for i in self.second_info:
            cur_info = defaultdict(float)
            for j in i:
                cur_info[to_cell(j[1], j[2])] += 1
            cur_sum = sum(cur_info.values())
            for j in cur_info:
                cur_info[j] /= cur_sum
            second_web_tf.append(cur_info)

        web_idf = defaultdict(float)
        for i in first_web_tf:
            for j in i:
                web_idf[j] += 1
        for i in second_web_tf:
            for j in i:
                web_idf[j] += 1
        sum_idf = sum(web_idf.values())
        for i in web_idf:
            web_idf[i] = math.log(sum_idf / web_idf[i])

        web_first = defaultdict(dict)
        web_second = defaultdict(dict)
        for i in range(len(first_web_tf)):
            for j in first_web_tf[i]:
                web_first[j][i] = web_first[j].get(i, 0) + first_web_tf[i][j] * web_idf[j]
        for i in range(len(second_web_tf)):
            for j in second_web_tf[i]:
                web_second[j][i] = web_second[j].get(i, 0) + second_web_tf[i][j] * web_idf[j]

        for i in web_idf:
            for j in web_first[i]:
                for k in web_second[i]:
                    self.init_first_second[j][k] += (web_first[i][j]+web_second[i][k]) / web_gap

    def theory_exact_analyse(self):
        first_theory_exact = 0
        second_theory_exact = 0
        for i in range(self.first_count):
            cur_order = sorted(range(self.second_count), key=lambda x: -self.init_first_second[i][x])
            for j in range(self.hits):
                if self.first_name[i] == self.second_name[cur_order[j]]:
                    first_theory_exact += 1
                    break
        for i in range(self.second_count):
            cur_order = sorted(range(self.first_count), key=lambda x: -self.init_first_second[x][i])
            for j in range(self.hits):
                if self.second_name[i] == self.first_name[cur_order[j]]:
                    second_theory_exact += 1
                    break
        print('theory exact rate: first %.4f, second %.4f'
              % (first_theory_exact/self.first_count, second_theory_exact/self.second_count))

    def make_loss(self):
        start_emb = tf.nn.embedding_lookup(self.first_emb, self.start_ph)
        end_emb = tf.nn.embedding_lookup(self.second_emb, self.end_ph)
        operation = tf.reduce_sum(tf.multiply(start_emb, end_emb), axis=1)
        operation = -tf.log_sigmoid(operation)
        return tf.reduce_sum(tf.multiply(operation, self.weight_ph))
        # return tf.reduce_mean(tf.multiply(operation, self.weight_ph))

    def make_model(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.loss)

    def run_train(self):
        self.save_init()

        start = []
        end = []
        weight = []
        for i in range(self.first_count):
            for j in range(self.second_count):
                start.append(i)
                end.append(j)
                weight.append(self.first_second[i][j])
        # print(start)
        # print(end)
        # print(weight)
        feed_dict = {self.start_ph: np.array(start),
                     self.end_ph: np.array(end),
                     self.weight_ph: np.array(weight)}

        train_time = time.time()
        for i in range(1, self.round + 1):
            self.session.run(self.train_model, feed_dict=feed_dict)
            if not i % self.stage:
                print('round: %s, train time: %.2f, loss: %.4f'
                      % (i, time.time()-train_time, self.session.run(self.loss, feed_dict=feed_dict)))
        print('train finish')
        self.save_res(self.session.run(self.loss, feed_dict=feed_dict))

    def save_init(self):
        emb_file = os.path.join(self.save_fold, 'embedding.txt')
        loss_file = os.path.join(self.save_fold, 'loss.txt')
        similar_file = os.path.join(self.save_fold, 'similarity.txt')
        for i in [emb_file, loss_file, similar_file]:
            with open(i, 'w')as f:
                f.write('')

    def save_res(self, loss=0):
        emb_file = os.path.join(self.save_fold, 'embedding.txt')
        loss_file = os.path.join(self.save_fold, 'loss.txt')
        similar_file = os.path.join(self.save_fold, 'similarity.txt')

        with open(emb_file, 'w')as f:
            first_emb = self.session.run(self.first_emb)
            second_emb = self.session.run(self.second_emb)
            f.write('first emb:\n')
            for i in range(self.first_count):
                f.write('%s: %s\n' % (self.first_name[i], first_emb[i]))
            f.write('\n')
            f.write('second emb:\n')
            for i in range(self.second_count):
                f.write('%s: %s\n' % (self.second_name[i], second_emb[i]))
            f.write('\n')

        with open(loss_file, 'w')as f:
            f.write(str(loss))

        with open(similar_file, 'w')as f:
            def make_similarity_matrix():
                print("learning rate：",self.learning_rate)
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
            first_exact = []
            for i in range(self.first_count):
                order = sorted(range(self.second_count), key=lambda x: -similarity_matrix[i][x])
                for j in range(self.second_count):
                    if self.first_name[i] == self.second_name[order[j]]:
                        first_exact.append(j + 1)
                        break
            second_exact = []
            for i in range(self.second_count):
                order = sorted(range(self.first_count), key=lambda x: -similarity_matrix[x][i])
                for j in range(self.first_count):
                    if self.second_name[i] == self.first_name[order[j]]:
                        second_exact.append(j + 1)
                        break
            print('exact time: %.2f' % (time.time() - sim_time))

            first_exact_rate = sum(1 for i in range(self.first_count) if first_exact[i] <= self.hits)
            first_exact_rate /= self.first_count
            second_exact_rate = sum(1 for i in range(self.second_count) if second_exact[i] <= self.hits)
            second_exact_rate /= self.second_count
            print('first exact rate: %.4f' % first_exact_rate)
            print('second exact rate: %.4f' % second_exact_rate)
            print('average_rate: %.4f'%(first_exact_rate/2+second_exact_rate/2))
            f.write(str(first_exact_rate) + '\n')
            f.write(str(second_exact_rate) + '\n\n')
            f.write('first_exact:\n')
            for i in first_exact:
                f.write(str(i) + '\n')
            f.write('second_exact:\n')
            for i in second_exact:
                f.write(str(i) + '\n')


def main(data_fold, save_fold):
    total_time = time.time()
    with  tf.Graph().as_default() ,tf.Session()as sess:
        ob = Model(data_fold, save_fold, sess,lr=0.01)
        ob.run_train()
    print('total time: %.4f' % (time.time()-total_time))
#[0.005,0.0075,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
#[0.8323.0.8262,0.8270,,0.8259,0.8553,0.8259,0.8238,0.8218,0.8212,0.8034,0.6904,0.4266,0.1945,0.0882,0.0550,0.0358,0.0236,0.0213,0.0143,0.0119,0.0119,0.0073]
if __name__ == '__main__':
    main(data_fold='Data', save_fold='Result')
