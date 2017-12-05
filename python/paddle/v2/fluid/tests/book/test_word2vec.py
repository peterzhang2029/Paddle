import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import time

PASS_NUM = 4
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32
IS_SPARSE = True

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_forth = fluid.layers.embedding(
    input=forth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')

concat_embed = fluid.layers.concat(
    input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
hidden1 = fluid.layers.fc(input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
avg_cost = fluid.layers.mean(x=cost)
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict_word, label=next_word)
train_reader = paddle.batch(
    paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(
    feed_list=[first_word, second_word, third_word, forth_word, next_word],
    place=place)
exe.run(fluid.default_startup_program())

for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    batch_id = 0
    print("pass: ", pass_id," begin")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for data in train_reader():
        avg_cost_np, acc = exe.run(fluid.default_main_program(),
                              feed=feeder.feed(data),
                              fetch_list=[avg_cost] + accuracy.metrics)
        pass_acc = accuracy.eval(exe)
        
        if batch_id % 100 == 0 and batch_id != 0:
            print("batch_id=" + str(batch_id) + " train_cost=" + str(avg_cost_np[0]) 
                  + " train_acc=" + str(acc) + " train_pass_acc=" + str(pass_acc))

        batch_id += 1
    print("pass: ", pass_id," end")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))