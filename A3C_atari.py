#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao62995@gmail.com

import cv2
import re
import gym
import signal
import threading
import scipy.signal
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from common import *

from parameters import *

from environment.AtariEnv import AtariEnv
from agent.A3CNet import A3CNet
from agent.A3CLSTMNet import A3CLSTMNet
from A3CSingleThread import A3CSingleThread

class A3CAtari(object):
    def __init__(self):
        self.env = AtariEnv(gym.make(flags.game))
        self.graph = tf.get_default_graph()
        # shared network
        if flags.use_lstm:
            self.shared_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim, scope="global_net")
        else:
            self.shared_net = A3CNet(self.env.state_shape, self.env.action_dim, scope="global_net")
        # shared optimizer
        self.shared_opt, self.global_step, self.summary_writer = self.shared_optimizer()
        # local training threads
        self.jobs = []
        for thread_id in xrange(flags.jobs):
            job = A3CSingleThread(thread_id, self)
            self.jobs.append(job)
        # session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                     allow_soft_placement=True))
        self.sess.run(tf.initialize_all_variables())
        # saver
        self.saver = tf.train.Saver(var_list=self.shared_net.get_vars(), max_to_keep=3)
        restore_model(self.sess, flags.train_dir, self.saver)
        self.global_step_val = 0

    def shared_optimizer(self):
        with tf.device("/gpu:%d" % flags.gpu):
            # optimizer
            if flags.opt == "rms":
                optimizer = tf.train.RMSPropOptimizer(flags.learn_rate, decay=0.99, epsilon=0.1, name="global_optimizer")
            elif flags.opt == "adam":
                optimizer = tf.train.AdamOptimizer(flags.learn_rate, name="global_optimizer")
            else:
                logger.error("invalid optimizer", to_exit=True)
            global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)
            summary_writer = tf.train.SummaryWriter(flags.train_dir, graph_def=self.graph)
        return optimizer, global_step, summary_writer

    def train(self):
        flags.train_step = 0
        signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

    def test_sync(self):
        self.env.reset_env()
        done = False
        map(lambda job: self.sess.run(job.sync), self.jobs)
        step = 0
        while not done:
            step += 1
            action = random.choice(range(self.env.action_dim))
            for job in self.jobs:
                pi = job.local_net.get_policy(self.sess, self.env.state)
                val = job.local_net.get_value(self.sess, self.env.state)
                _, _, done = self.env.forward_action(action)
                print "step:", step, ", job:", job.thread_id, ", policy:", pi, ", value:", val
            print
        print "done!"

    def test(self):
        self.env.reset_env()
        self.saver.restore(self.sess, "models/experiment0/a3c_model-2533")
        done = False
        map(lambda job: self.sess.run(job.sync), self.jobs)
        step = 0
        self.jobs[0].forward_explore(10000)
        print "done!"

def signal_handler():
    sys.exit(0)


def main(_):
    # mkdir
    if not os.path.isdir(flags.train_dir):
        os.makedirs(flags.train_dir)
    # remove old tfevents files
    for f in os.listdir(flags.train_dir):
        if re.search(".*tfevents.*", f):
            os.remove(os.path.join(flags.train_dir, f))
    # model
    model = A3CAtari()
    # model.train()
    model.test()

if __name__ == "__main__":
    tf.app.run()
