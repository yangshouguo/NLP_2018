
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


    def load_data(self, filename, rate = 0.2, seg = 0):#rate 表示测试集所占的比例
        fp = open(filename, 'r', encoding = 'utf-8')
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

    def _generate_tag(self, length, TF, pre = 'S'):

        if length == 0:
            return '\n'

        if TF:
            tag = [pre+'I' for i in range(length)]
            tag[0] = pre+'B'
            tag[-1] = pre+'E'
            return tag
        else:
            return ['S' for i in range(length)]

    #将数据集单个字进行标注
    # 特征 前一个词是 nnt/10414 w/22441 v/4611       n/3246 ude1/2433 p/2721
    # 特征 后一个词是 v/13272 d/4882 w/14238         p/3694 n/4827 ude1/3481  n/4872
    def process_single_char(self, sentences, cx):
        nnt = []
        w = []
        v_n = []
        d = []
        tag = [] # nr
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                nnt.append(self._generate_tag(len(sentences[i][j]), cx[i][j] == 'nnt', pre='NNT_'))
                w.append(self._generate_tag(len(sentences[i][j]), cx[i][j] == 'w', pre='W_'))
                v_n.append(self._generate_tag(len(sentences[i][j]), cx[i][j] == 'v', pre='V_'))
                d.append(self._generate_tag(len(sentences[i][j]), cx[i][j] == 'd',pre = 'D_'))
                tag.append(self._generate_tag(len(sentences[i][j]), cx[i][j] == 'nr',pre='NR_'))

        return (sentences, nnt, w, v_n, d, tag)

    #将数据集单个字进行标注
    # 特征 前一个词是 nnt/10414 w/22441 v/4611       n/3246 ude1/2433 p/2721
    # 特征 后一个词是 v/13272 d/4882 w/14238         p/3694 n/4827 ude1/3481  n/4872
    # 将特征放在同一列
    def process_single_char2(self, sentences, cx):
        features = []
        tag = [] # nr
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if cx[i][j] == 'notw':
                    features.append(self._generate_tag(len(sentences[i][j]), True, pre='W_'))
                elif cx[i][j] == 'nnt':
                    features.append(self._generate_tag(len(sentences[i][j]), True, pre='NNT_'))
                elif cx[i][j] == 'v':
                    features.append(self._generate_tag(len(sentences[i][j]), True, pre='V_'))
                elif cx[i][j] == 'd':
                    features.append(self._generate_tag(len(sentences[i][j]), True, pre='D_'))
                else:
                    features.append(self._generate_tag(len(sentences[i][j]), False, pre='S_'))

                tag.append(self._generate_tag(len(sentences[i][j]), cx[i][j] == 'nr',pre='NR_'))

        return (sentences, features, tag)
    #统计
    def statistic(self, sentences, states):
        pre = {}
        next = {}
        for i in range(len(states)):
            for j in range(len(states[i])):
                if 'nr' in states[i][j]:
                    if j != 0 :
                        if states[i][j-1] not in pre:
                            pre[states[i][j-1]] = 0

                        pre[states[i][j-1]] += 1
                    if j!= len(states[i])-1 :
                        if states[i][j+1] not in next:
                            next[states[i][j+1]] = 0
                        next[states[i][j+1]] += 1

        return pre,next


    #生成交叉验证的训练集和测试集
    def generate_data(self, ord_i):

        (train_o, train_i), (test_o, test_i) = self.load_data(self._filename, seg=ord_i)
        #(sentences, nnt, w, v_n, d, tag) = self.process_single_char(train_i, train_o)
        (sentences, features, tag) = self.process_single_char2(train_i, train_o)
        with open(str(ord_i)+'train_data.txt', 'w') as f:
            pos = 0
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    for k in range(len(sentences[i][j])):
                        line = sentences[i][j][k]+' '+features[pos][k]+' '+tag[pos][k]+'\n'
                        f.write(line)
                    pos += 1

                f.write('\n')
        (sentences, features, tag) = self.process_single_char2(test_i, test_o)
        with open(str(ord_i)+'test_data.txt', 'w') as f:
            pos = 0
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    for k in range(len(sentences[i][j])):
                        line = sentences[i][j][k]+' '+features[pos][k]+' '+tag[pos][k]+'\n'
                        f.write(line)
                    pos += 1

                f.write('\n')


if __name__ == '__main__':
    model = Model('raw_data.txt')

    for i in range(5):
        print('generating %d th data set......' % i)
        model.generate_data(i)

