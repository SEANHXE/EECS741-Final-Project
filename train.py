from __future__ import print_function
import os
import paddle
import paddle.fluid as fluid
import numpy
import sys

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=False):
    tmp= fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)


def shortcut(input, ch_in, ch_out, stride):
    if ch_in!= ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_in, ch_out, stride):
    tmp= conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp= conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short= shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp= block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp= block_func(tmp, ch_out, ch_out, 1)
    return tmp



def resnet_lfw(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth- 2) % 6== 0
    n= (depth- 2) // 6
    nStages= {16, 64, 128}
    conv1= conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
    res1= layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2= layer_warp(basicblock, res1, 16, 32, n, 2)
    res3= layer_warp(basicblock, res2, 32, 64, n, 2)
    pool= fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict= fluid.layers.fc(input=pool, size=10, act='softmax')
    return predict



def inference_network():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')


    predict = resnet_lfw(images, 32)
    #predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict


def train_network(predict):
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


# Each batch will yield 128 images
BATCH_SIZE= 128

# Reader for training
train_reader = paddle.batch(
    paddle.reader.shuffle(
       paddle.dataset.lfw.train10(), buf_size=128 * 100),
    batch_size=BATCH_SIZE)
# Reader for testing. A separated data set for testing.
test_reader = paddle.batch(
   paddle.dataset.lfw.test10(), batch_size=BATCH_SIZE)

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

feed_order = ['pixel', 'label']

main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()

predict = inference_network()
avg_cost, acc = train_network(predict)

# Test program
test_program = main_program.clone(for_test=True)

optimizer = optimizer_program()
optimizer.minimize(avg_cost)

exe = fluid.Executor(place)

EPOCH_NUM = 10
# For training test cost
def train_test(program, reader):
    count = 0
    feed_var_list = [
       program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, acc]) * [0]
    for tid, test_data in enumerate(reader()):
        avg_cost_np = test_exe.run(
            program=program,
           feed=feeder_test.feed(test_data),
           fetch_list=[avg_cost, acc])
        accumulated = [
            x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
        ]
        count += 1
    return [x / count for x in accumulated]


params_dirname = "image_classification_resnet.inference.model"

from paddle.utils.plot import Ploter

train_prompt = "Train cost"
test_prompt = "Test cost"
plot_cost = Ploter(test_prompt,train_prompt)

def event_handler_plot(ploter_title, step, cost):
    plot_cost.append(ploter_title, step, cost)
    plot_cost.plot()


# main train loop.
def train_loop():
    feed_var_list_loop = [
       main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
    exe.run(star_program)

    step = 0
    EPOCH_NUM = 20
    for pass_id in range(EPOCH_NUM):
        for step_id, data_train in enumerate(train_reader()):
            avg_loss_value = exe.run(
                main_program,
               feed=feeder.feed(data_train),
               fetch_list=[avg_cost, acc])
            if step_id % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, Acc %f" % (pass_id, step_id, avg_loss_value[0], avg_loss_value[1]))
                event_handler_plot(train_prompt, step, avg_loss_value[0])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
            step += 1

    avg_cost_test, accuracy_test = train_test(
        test_program, reader=test_reader)
    print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
        pass_id, avg_cost_test, accuracy_test))
    event_handler_plot(test_prompt, step, avg_cost_test)

    plot_cost.plot()
'''
    if params_dirname is not None:
        predict = inference_network()
        fluid.io.save_inference_model(params_dirname, ["pixel"], [predict], exe)
'''

train_loop()
