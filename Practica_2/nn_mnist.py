import gzip
import cPickle

import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
from samba.dcerpc.base import transfer_syntax_ndr



def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)

f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784]) # 28x28
y_ = tf.placeholder("float", [None, 10]) # from 0 to 9

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
old_error = 10000
epoch = 0

#for epoch in xrange(100):
while True:
    epoch+=1
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    errore = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    print "Epoch #:", epoch, "Error: ", errore, "NewErr - OldErr: ", old_error-errore
    result = sess.run(y, feed_dict={x: batch_xs})
    if old_error - errore < 0.1:
        break
    old_error = errore;

    #for b, r in zip(batch_ys, result):
    #    print b, "-->", r
    #print "----------------------------------------------------------------------------------"

result = sess.run(y, feed_dict= {x:test_x})
errors = 0
for b,r in zip(test_y, result):
    b = np.argmax(b)
    r = np.argmax(r)

    if b != r:
        print b, "----->",r, "   Error"
        errors += 1
    else:
        print b, "----->",r,"   OK"

print "\nTotal errors: " , errors, "/", len(test_y), "Accuracy:", (len(test_y)-errors)*100.0/len(test_y), "%"
