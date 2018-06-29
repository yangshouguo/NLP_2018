
import csv
import random
import numpy as np
import tensorflow as tf
# from ffnn.FFNN_MODEL import model


n_class = 3
alldata = []
input = []
output = []

with open("./wine.all.txt", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    alldata = [list(map(float, row)) for row in reader]


np.random.shuffle(alldata)

alldata = np.array(alldata)

input = alldata[:, 1:]
output = alldata[:, 0]
encode_output = list(map(int, output))

encode_output = tf.one_hot(np.array(encode_output)-1, depth=3)
with tf.Session() as sess:
    encode_output = sess.run(encode_output)

train_size = (int)(len(input) * 0.8)

train_input = input[0:train_size]
train_output = encode_output[0:train_size]

test_input = input[train_size:]
test_output = encode_output[train_size:]

def add_layer(inputs, in_size, out_size, activation_function = None, keep_p = 0):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, W) + biases

    if keep_p != 0:
        Wx_plus_b_drop = tf.nn.dropout(Wx_plus_b, keep_p)
    else:
        Wx_plus_b_drop = Wx_plus_b

    if activation_function is None:
        return Wx_plus_b_drop
    else:
        return activation_function(Wx_plus_b_drop)

def compute_accuracy(x, y):
    global predict
    pred = sess.run(predict, feed_dict={xs:x})
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:x, ys:y})

    return result


xs = tf.placeholder(tf.float32, [None, 13])
ys = tf.placeholder(tf.float32, [None, 3])

h_layer = add_layer(xs, 13, 600, activation_function = tf.nn.sigmoid)

hidden_layer = add_layer(h_layer, 600, 400, activation_function=tf.nn.sigmoid, keep_p=0.9)

predict = add_layer(hidden_layer, 400, 3, activation_function=None)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=ys)
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predict),
#                                reduction_indices=[1]))

lr = 0.5 # learning rate

train_step = tf.train.AdadeltaOptimizer(lr).minimize(tf.reduce_mean(loss))



epoch = 2000
batchsize = 20


sess = tf.Session()

sess.run(tf.global_variables_initializer())
# sess.run(encode_output)
for j in range(epoch):
    for i in range(int(len(train_input)/batchsize)):
        sess.run(train_step, feed_dict={xs: train_input[i*batchsize:(i+1)*batchsize].reshape(-1, 13), ys : train_output[i*batchsize:(i+1)*batchsize].reshape(-1, 3)})
    if len(train_input) % batchsize != 0:
        sess.run(train_step, feed_dict={xs: train_input[-(len(train_input)%batchsize):].reshape(-1, 13), ys : train_output[-(len(train_input)%batchsize):].reshape(-1, 3)})
    if j % 100 == 0:

        # pre = sess.run(predict, feed_dict={xs:train_input, ys:train_output})
        print('epoch %d' % j)

        print('loss : %f' % sess.run(tf.reduce_mean(loss), feed_dict={xs: train_input, ys: train_output}))

        print('accuracy : %f' % (compute_accuracy(train_input, train_output)))

        print('test acc %f !' % (compute_accuracy(test_input, test_output)))

        print()

saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './model')

print('test data accuracy %f !' % (compute_accuracy(test_input, test_output)))