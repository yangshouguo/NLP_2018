import numpy as np
import copy
import math
class Model:

    def __init__(self, filename):


        self._filename = filename #语料文件
        self._beginer = {} #统计以词性x开头的次数
        self._allSentence = 0 #总的句子数量
        self._transform = {} #词性转移矩阵  transform[i][j]表示词性i转移到j的次数 transform[i]['count'] 表示所有从词性i转移到其他词性的次数
        self._transform_vocabulary = {} #词性到单词的统计  transform_vocabulary[i][v]表示从词性i输出到单词v的次数,transform_vocabulary[i]['count']表示词性i出现的总次数
        self._acc = []
        self._pi = {}
        self._A = {}
        self._B = {}
        pass

    def _var_init(self):
        self._S = set()
        self._K = set()


    def load_data(self, filename, rate = 0.2, seg = 0):#rate 表示测试集所占的比例
        fp = open(filename, 'r')
        lines = fp.readlines()
        data_len = len(lines)
        test_size = (int)(data_len*(rate))

        start = test_size * seg

        test_set = lines[start:start+test_size]
        training_set = lines[0:start] + lines[start+test_size:]

        # #随机分割
        # train_size = data_len - test_size
        # np.random.shuffle(lines)
        # training_set, test_set = lines[0:train_size], lines[train_size:]

        return self._preprocess(training_set),self._preprocess(test_set)

    def _preprocess(self, dataset_row):
        input = []
        output = []

        sentence = 0
        for line in dataset_row:
            input.append([])
            output.append([])
            words= line.rstrip('\n').split(' ')
            for word in words:
                if len(word)<1:
                    continue
                if '/' not in word:
                    continue
                wv = self._split_from_end(word)

                if len(wv) < 2 or len(wv[1]) < 1:
                    continue

                if not wv[1][0].isalpha():
                    print(wv)
                input[sentence].append(wv[0])
                output[sentence].append(wv[1])

            sentence += 1

        return output, input

    #从最后一个分割符进行分割
    # 例如 1/2/m 分割之后 1/2 , m
    def _split_from_end(self, str, spliter = '/'):

        pos = str.rfind(spliter)
        return str[:pos],str[pos+1:]

    def _add_to_dict(self, dict, key, value = 0):

        if key not in dict:
            dict[key] = value

    def train(self, seg = 0):

        self._var_init()

        (self._train_x , self._train_y), (self._test_x, self._test_y) = self.load_data(self._filename, rate=0.2, seg=seg)

        train_size = len(self._train_x)
        self._allSentence = train_size


        for i in range(train_size):
            voc = self._train_x[i][0]
            if voc not in self._beginer:
                self._beginer[voc] = 1
            else:
                self._beginer[voc] += 1

            for j in range(len(self._train_x[i]) - 1):

                #分割错误问题，已经解决 _split_from_end
                # if not self._train_x[i][j][0].isalpha():
                #     print(self._train_x[i])
                #     print(self._train_y[i])
                #     print(self._train_x[i][j])
                #     print(self._train_y[i][j])

                self._add_to_dict(self._transform, self._train_x[i][j], {})
                self._add_to_dict(self._transform[self._train_x[i][j]], self._train_x[i][j+1], 0)
                self._add_to_dict(self._transform[self._train_x[i][j]], 'count', 0)
                self._transform[self._train_x[i][j]][self._train_x[i][j+1]] += 1
                self._add_to_dict(self._transform_vocabulary, self._train_x[i][j], {})
                self._add_to_dict(self._transform_vocabulary[self._train_x[i][j]], self._train_y[i][j], 0)
                self._add_to_dict(self._transform_vocabulary[self._train_x[i][j]], 'count', 0)
                self._transform_vocabulary[self._train_x[i][j]][self._train_y[i][j]] += 1
                self._transform_vocabulary[self._train_x[i][j]]['count'] += 1
                self._transform[self._train_x[i][j]]['count'] += 1

            self._add_to_dict(self._transform_vocabulary, self._train_x[i][-1], {})
            self._add_to_dict(self._transform_vocabulary[self._train_x[i][-1]],self._train_y[i][-1], 0)
            self._transform_vocabulary[self._train_x[i][-1]][self._train_y[i][-1]] += 1
            self._add_to_dict(self._transform_vocabulary[self._train_x[i][-1]], 'count', 0)
            self._transform_vocabulary[self._train_x[i][-1]]['count'] += 1

        self._compute_prob()

    def _compute_prob(self):#根据统计数据计算概率值

        for key in self._beginer:
            self._pi[key] = self._beginer[key] / self._allSentence

        for key in self._transform:

            for trans_key in self._transform[key]:
                self._add_to_dict(self._A, key, {})
                self._A[key][trans_key] = (self._transform[key][trans_key]+1) / (self._transform[key]['count']+1)


        for key in self._transform_vocabulary:
            if key != '':
                self._S.add(key)
            for word in self._transform_vocabulary[key]:
                self._K.add(word)
                self._add_to_dict(self._B, key, {})
                self._B[key][word] = (self._transform_vocabulary[key][word]+1) / (self._transform_vocabulary[key]['count']+1)
        self._S = list(self._S)
        self._K = list(self._K)


    def _pre_one(self, Oseq):
        #Vertibi algorithm
        #init
        dp_t = {}
        dp_t1 = {}
        record = [] # 记录到达路径

        for i in range(len(self._S)):
            if self._S[i] not in self._pi:
                dp_t[self._S[i]] = math.log(1/self._allSentence)
                continue

            if Oseq[0] in self._B[self._S[i]]:
                dp_t[self._S[i]] = math.log(self._pi[self._S[i]]) + math.log(self._B[self._S[i]][Oseq[0]])
            else:
                dp_t[self._S[i]] = math.log(self._pi[self._S[i]])+ math.log(1 / (self._transform_vocabulary[self._S[i]]['count'] + 1)) #平滑

        #viterbi算法核心,dp
        for i in range(1,len(Oseq)):
            change = {}
            for j in range(len(self._S)):
                Out = Oseq[i]
                hidden_state = self._S[j]
                dp_t1[self._S[j]], laststate = self._maxS(dp_t, Out, hidden_state)
                change[self._S[j]] = laststate

            dp_t = copy.deepcopy(dp_t1)
            record.append(change)
        final_state = ''
        final_score = - math.inf
        for i in range(len(self._S)):
            if dp_t[self._S[i]] > final_score:
                final_score = dp_t[self._S[i]]
                final_state = self._S[i]

        states = []
        states.append(final_state)
        for i in range(len(record)-1,-1,-1):
            states.append(record[i][final_state])
            final_state = record[i][final_state]
        states.reverse()
        return states

    def _maxS(self, dp_t, O, state):

        score_state = - math.inf
        last_state = ''
        for i in range(len(self._S)):

            if state not in self._A[self._S[i]]:
                continue


            if dp_t[self._S[i]]+math.log(self._A[self._S[i]][state]) > score_state:
                score_state = dp_t[self._S[i]]+math.log(self._A[self._S[i]][state])
                last_state = self._S[i]

        if O not in self._B[state]:
            return score_state+math.log(0.00000001),last_state

        return score_state+math.log(self._B[state][O]), last_state



    def predict(self):
        all_word = 0 #总词数
        match_word = 0 #正确的词数
        for i in range(len(self._test_y)):
            state = self._pre_one(self._test_y[i])
            print('待预测的句子',self._test_y[i])
            print('长度',len(self._test_y[i]))
            print('预测：',state)
            print('长度:',len(state))
            print('实际',self._test_x[i])
            l = len(state)
            all_word += l
            for j in range(l):
                if state[j] == self._test_x[i][j]:
                    match_word += 1

            print('总词数', all_word, '正确预测词性的数量',match_word )
            print('accuracy', match_word/all_word)

        self._acc.append(match_word/all_word)

    def summary(self):

        print(self._acc)
        sum = 0
        for a in self._acc:
            sum+= a

        print('ave acc:', sum/len(self._acc))

if __name__ == '__main__':
    model = Model('../raw_data.txt')

    for i in range(5):
        model.train(i)
        model.predict()

    model.summary()

