import numpy as np
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('os.environ:', os.environ['CUDA_VISIBLE_DEVICES'])
import tensorflow as tf

# tf.reset_default_graph()
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
from model_sparse import SDPP
import sys
import six.moves.cPickle as pickle
import pickle
import gzip

tf.set_random_seed(0)
import time
import config as timecas_config

NUM_THREADS = 20
# DATA_PATH = "data"

print('ok')

print(timecas_config.information)
n_nodes, n_sequences, n_steps = pickle.load(open(timecas_config.information, 'rb'))
print("dataset information: nodes:{}, n_sequence:{}, n_steps:{} ".format(n_nodes, n_sequences, n_steps))
tf.flags.DEFINE_integer("n_sequences", n_sequences, "num of sequences.")
tf.flags.DEFINE_integer("n_steps", n_steps, "num of step.")
tf.flags.DEFINE_integer("time_interval", timecas_config.time_interval, "the time interval")
tf.flags.DEFINE_integer("n_time_interval", timecas_config.n_time_interval, "the number of  time interval")
print(len(sys.argv))
learning_rate, emb_learning_rate = 0.005, 0.0005
# learning_rate, emb_learning_rate = 0.005, 0.005
# learning_rate, emb_learning_rate = 0.003, 0.003


l2, dropout = 0.05, 1.

if len(sys.argv) > 1:
    learning_rate = float(sys.argv[1])
    emb_learning_rate = float(sys.argv[2])
    l2 = float(sys.argv[3])
    dropout = float(sys.argv[4])

tf.flags.DEFINE_float("learning_rate", learning_rate, "learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", 20, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size", 32, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", 32, "hidden gru size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", l2, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("training_iters", 200 * 3200 + 1, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")

tf.flags.DEFINE_integer("embedding_size", 50, "embedding size.")
tf.flags.DEFINE_integer("n_input", 50, "input size.")

# tf.flags.DEFINE_integer("embedding_size", 128, "embedding size.")
# tf.flags.DEFINE_integer("n_input", 128, "input size.")

tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 32, "dense2 size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", emb_learning_rate, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", dropout, "dropout probability.")

tf.flags.DEFINE_boolean("PRETRAIN", False, "Loading PRETRAIN models or not.")

tf.flags.DEFINE_boolean("fix", False, "Fix the pretrained embedding or not.")


tf.flags.DEFINE_boolean("classification", False, "classification or regression.")
tf.flags.DEFINE_integer("n_class", 5, "number of class if do classification.")

tf.flags.DEFINE_boolean("one_dense_layer", False, "number of dense layer out output.")

config = tf.flags.FLAGS

print("dropout prob:", config.dropout_prob)
print("l2", config.l2)
print("learning rate:", config.learning_rate)
print("emb_learning_rate:", config.emb_learning_rate)

def get_y_label(batch_y):
    y_label = np.zeros_like(batch_y)
    y_label[(batch_y>0) & (batch_y <= np.log(10.+1.))] = 0
    batch_y[(batch_y>0) & (batch_y <= np.log(10.+1.))] = 0
    y_label[(batch_y>0) & (batch_y <= np.log(50.+1.))] = 1
    batch_y[(batch_y>0) & (batch_y <= np.log(50.+1.))] = 0
    y_label[(batch_y>0) & (batch_y <= np.log(100.+1.))] = 2
    batch_y[(batch_y>0) & (batch_y <= np.log(100.+1.))] = 0
    y_label[(batch_y>0) & (batch_y <= np.log(500.+1.))] = 3
    batch_y[(batch_y>0) & (batch_y <= np.log(500.+1.))] = 0
    y_label[batch_y>0] = 4
    return y_label

# (total_number of sequence,n_steps)
def get_batch(x, y, sz, time, rnn_index, n_time_interval, step, batch_size=128):
    batch_y = np.zeros(shape=(batch_size, 1))
    batch_x = []
    batch_x_indict = []
    batch_time_interval_index = []
    batch_rnn_index = []

    start = step * batch_size % len(x)
    # print start
    for i in range(batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        for j in range(sz[id]):
            batch_x.append(x[id][j])
            # time_interval
            temp_time = np.zeros(shape=(n_time_interval))
            k = int(math.floor(time[id][j] / config.time_interval))
            # in observation_num model, the k can be larger than n_time_interval
            if k >= config.n_time_interval:
                k = config.n_time_interval - 1

            temp_time[k] = 1
            batch_time_interval_index.append(temp_time)

            # rnn index
            temp_rnn = np.zeros(shape=(config.n_steps))
            if rnn_index[id][j] - 1 >= 0:
                temp_rnn[rnn_index[id][j] - 1] = 1
            batch_rnn_index.append(temp_rnn)

            for k in range(2 * config.n_hidden_gru):
                batch_x_indict.append([i, j, k])

    if config.classification:
        batch_y = get_y_label(batch_y)

    return batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index


version = config.version
x_train, y_train, sz_train, time_train, rnn_index_train, vocabulary_size = pickle.load(
    open(timecas_config.train_pkl, 'rb'))
x_test, y_test, sz_test, time_test, rnn_index_test, _ = pickle.load(open(timecas_config.test_pkl, 'rb'))
x_val, y_val, sz_val, time_val, rnn_index_val, _ = pickle.load(open(timecas_config.val_pkl, 'rb'))

# do analysis
def analysis_data(Y_train, Y_test, Y_valid):
    print('---------***---------')
    print("Number: {}, Max: {} and Min: {} label Value in Train".format(len(Y_train), max(Y_train), min(Y_train)))
    print('NUmber: {}, Max: {} and Min: {} label Value in Test'.format(len(Y_test), max(Y_test), min(Y_test)))
    print('NUmber: {}, Max: {} and Min: {} label Value in Valid'.format(len(Y_valid), max(Y_valid), min(Y_valid)))
    print('---------***---------')

analysis_data(y_train, y_test, y_val)
print(len(x_train), len(x_test), len(x_val))

training_iters = config.training_iters
batch_size = config.batch_size
# display_step = min(config.display_step, len(sz_train) / batch_size)
# display_step = 5
display_step = math.ceil(len(x_train)/batch_size) # display for each epoch
# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = SDPP(config, sess, n_nodes)
# sess.graph.finalize()
# sess.run(tf.global_variables_initializer())


step = 0
best_val_loss = 1000
best_test_loss = 1000

train_writer = tf.summary.FileWriter("./train", sess.graph)

# Keep training until reach max iterations or max_try
train_loss = []
train_mse = []
max_try = 20
patience = max_try
import time
start = time.time()
print(math.ceil(len(sz_train)/batch_size))
# 保存模型
saver = tf.train.Saver(max_to_keep=5)

while step * batch_size < training_iters:
    batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index = get_batch(x_train, y_train, sz_train,
                                                                                             time_train,
                                                                                             rnn_index_train,
                                                                                             config.n_time_interval,
                                                                                             step,
                                                                                             batch_size=batch_size)
    lambda1 = model.train_batch(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)
    # saver = tf.train.Saver()
    # saver.save(model.sess, '../save_models/train_model_%s.ckpt'%step)
    # train_writer.add_summary(summary, step)
    # preds = model.predict(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)

    embeddings2 = model.get_embedding2()
    loss_train = model.get_error(batch_x, batch_x_indict, batch_y, batch_time_interval_index, batch_rnn_index)
    train_loss.append(loss_train)

    # saver.save(sess, '../checkpoints/hawkes/hawkes.ckpt', global_step=100)
    if (step + 1) % display_step == 0:
        saver.save(sess, '../checkpoints/hawkes/hawkes.ckpt', global_step=step)

    if (step+1) % display_step ==0 :
        print('finish one epoch, time{}'.format(time.time()-start))

    if (step+1) % display_step == 0:
        # Calculate batch loss
        # print lambda1

        train_loss = []
        train_pred, train_truth = [], []
        for train_step in range(int(len(y_train) / batch_size)):
            val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index = get_batch(x_train, y_train, sz_train,
                                                                                           time_train, rnn_index_train,
                                                                                           config.n_time_interval,
                                                                                           train_step,
                                                                                           batch_size=batch_size)
            train_loss.append(model.get_error(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index))

            predictions = model.predict(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index)
            if config.classification:
                predictions = predictions.argmax(axis=-1)
            train_pred.extend(predictions.squeeze().tolist())
            train_truth.extend(val_y.squeeze().tolist())

        print('train acc: ', np.sum(np.equal(np.array(train_pred), np.array(train_truth))) / len(train_pred))

        print(lambda1)
        val_loss = []
        val_pred, val_truth = [], []
        for val_step in range(int(len(y_val) / batch_size)):
            val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index = get_batch(x_val, y_val, sz_val,
                                                                                           time_val, rnn_index_val,
                                                                                           config.n_time_interval,
                                                                                           val_step,
                                                                                           batch_size=batch_size)
            val_loss.append(model.get_error(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index))

            predictions = model.predict(val_x, val_x_indict, val_y, val_time_interval_index, val_rnn_index)
            if config.classification:
                predictions = predictions.argmax(axis=-1)
            val_pred.extend(predictions.squeeze().tolist())
            val_truth.extend(val_y.squeeze().tolist())


        test_loss = []
        test_mse = []
        test_pred, test_truth = [], []
        for test_step in range(int(len(y_test) / batch_size)):
            test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_test, y_test, sz_test,
                                                                                                time_test,
                                                                                                rnn_index_test,
                                                                                                config.n_time_interval,
                                                                                                test_step,
                                                                                                batch_size=batch_size)
            test_loss.append(model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
            predictions = model.predict(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index)
            if config.classification:
                predictions = predictions.argmax(axis=-1)
            test_pred.extend(predictions.squeeze().tolist())
            test_truth.extend(test_y.squeeze().tolist())


        predict_result = []
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            patience = max_try

        if np.mean(test_loss) < best_test_loss:
            best_test_loss = np.mean(test_loss)

            # test_loss = []
            # for test_step in range(int(len(y_test) / batch_size + 1)):
            #     test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index = get_batch(x_test, y_test,
            #                                                                                         sz_test, time_test,
            #                                                                                         rnn_index_test,
            #                                                                                         config.n_time_interval,
            #                                                                                         test_step,
            #                                                                                         batch_size=batch_size)
            #     predict_result.extend(
            #         model.predict(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
            #     test_loss.append(
            #         model.get_error(test_x, test_x_indict, test_y, test_time_interval_index, test_rnn_index))
            #
            # print("last test error:", np.mean(test_loss))
            # pickle.dump((predict_result, y_test, test_loss), open(
            #     "prediction_result_total_l" + str(learning_rate) + "_embl" + str(emb_learning_rate) + "_l2" + str(
            #         l2) + "_dropout" + str(dropout) + ".pkl", 'wb'))
        if not config.classification:
            median_loss = np.median((np.array(test_pred)-np.array(test_truth))**2)
            median_val_loss = np.median((np.array(val_pred)-np.array(val_truth))**2)
        else:
            # acc in actually
            acc_test = np.sum(np.equal(np.array(test_pred), np.array(test_truth))) / len(test_pred)
            acc_valid = np.sum(np.equal(np.array(val_pred), np.array(val_truth))) / len(val_pred)

        if not config.classification:
            print("#" + str(int(step / display_step)) +
              ", Training Loss= " + "{:.5f}".format(np.mean(train_loss)) +
              ", Valid Loss= " + "{:.5f}".format(np.mean(val_loss)) +
              ", Valid median Loss= " + "{:.5f}".format(median_val_loss) +
              ", Best Valid Loss= " + "{:.5f}".format(best_val_loss) +
              ", Test Loss= " + "{:.5f}".format(np.mean(test_loss)) +
              ", Test median  Loss= " + "{:.5f}".format(median_loss) +
              ", Best Test Loss= " + "{:.5f}".format(best_test_loss)
              )
        else:
            print("#" + str(int(step / display_step)) +
                  ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) +
                  ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) +
                  ", Valid acc= " + "{:.6f}".format(acc_valid) +
                  ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
                  ", Test acc= " + "{:.6f}".format(acc_test) +
                  ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) +
                  ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
                  )
        # print("#" + str(step / display_step) +
        #       ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) +
        #       ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) +
        #       ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
        #       ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) +
        #       ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
        #       )
        train_loss = []
        patience -= 1
        if not patience:
            break
        # if best_val_loss <2.35:
    #    break
    step += 1


# saver.restore(sess, '../checkpoints/hawkes/hawkes.ckpt')


# print len(predict_result),len(y_test)
# print "Finished!\n----------------------------------------------------------------"
# print "Time:", time.time()-start
# print "Valid Loss:", best_val_loss
# print "Test Loss:", best_test_loss
# print lambda1
