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


class A3CSingleThread(threading.Thread):
    def __init__(self, thread_id, master):
        self.thread_id = thread_id
        threading.Thread.__init__(self, name="thread_%d" % thread_id)
        self.env = AtariEnv(gym.make(flags.game))
        self.master = master
        # local network
        if flags.use_lstm:
            self.local_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim, scope="local_net_%d" % thread_id)
        else:
            self.local_net = A3CNet(self.env.state_shape, self.env.action_dim, scope="local_net_%d" % thread_id)
        # sync network
        self.sync = self.sync_network(master.shared_net)
        # accumulate gradients
        self.accum_grads = self.create_accumulate_gradients()
        self.do_accum_grads_ops = self.do_accumulate_gradients()
        self.reset_accum_grads_ops = self.reset_accumulate_gradients()
        # collect summaries for debugging
        summaries = list()
        summaries.append(tf.scalar_summary("entropy/%d" % self.thread_id, self.local_net.entropy))
        summaries.append(tf.scalar_summary("policy_loss/%d" % self.thread_id, self.local_net.policy_loss))
        summaries.append(tf.scalar_summary("value_loss/%d" % self.thread_id, self.local_net.value_loss))
        summaries.append(tf.scalar_summary("total_loss/%d" % self.thread_id, self.local_net.total_loss))
        # apply accumulated gradients
        with tf.device("/gpu:%d" % flags.gpu):
            clip_accum_grads = [tf.clip_by_value(grad, -flags.grad_clip, flags.grad_clip) for grad in self.accum_grads]
            self.apply_gradients = master.shared_opt.apply_gradients(
                zip(clip_accum_grads, master.shared_net.get_vars()), global_step=master.global_step)
            self.summary_op = tf.merge_summary(summaries)

    def sync_network(self, source_net):
        sync_ops = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([], name="sync_ops_%d" % self.thread_id):
                for (target_var, source_var) in zip(self.local_net.get_vars(), source_net.get_vars()):
                    ops = tf.assign(target_var, source_var)
                    sync_ops.append(ops)
                return tf.group(*sync_ops, name="sync_group_%d" % self.thread_id)

    def create_accumulate_gradients(self):
        accum_grads = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([self.local_net], name="create_accum_%d" % self.thread_id):
                for var in self.local_net.get_vars():
                    zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                    name = var.name.replace(":", "_") + "_accum_grad"
                    accum_grad = tf.Variable(zero, name=name, trainable=False)
                    accum_grads.append(accum_grad.ref())
                return accum_grads

    def do_accumulate_gradients(self):
        net = self.local_net
        accum_grad_ops = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([net], name="grad_ops_%d" % self.thread_id):
                var_refs = [v.ref() for v in net.get_vars()]
                grads = tf.gradients(net.total_loss, var_refs, gate_gradients=False,
                                     aggregation_method=None,
                                     colocate_gradients_with_ops=False)
            with tf.op_scope([], name="accum_ops_%d" % self.thread_id):
                for (grad, var, accum_grad) in zip(grads, net.get_vars(), self.accum_grads):
                    name = var.name.replace(":", "_") + "_accum_grad_ops"
                    accum_ops = tf.assign_add(accum_grad, grad, name=name)
                    accum_grad_ops.append(accum_ops)
                return tf.group(*accum_grad_ops, name="accum_group_%d" % self.thread_id)

    def reset_accumulate_gradients(self):
        net = self.local_net
        reset_grad_ops = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([net], name="reset_grad_ops_%d" % self.thread_id):
                for (var, accum_grad) in zip(net.get_vars(), self.accum_grads):
                    zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                    name = var.name.replace(":", "_") + "_reset_grad_ops"
                    reset_ops = tf.assign(accum_grad, zero, name=name)
                    reset_grad_ops.append(reset_ops)
                return tf.group(*reset_grad_ops, name="reset_accum_group_%d" % self.thread_id)

    def weighted_choose_action(self, pi_probs):
        r = random.uniform(0, sum(pi_probs))
        upto = 0
        for idx, prob in enumerate(pi_probs):
            if upto + prob >= r:
                return idx
            upto += prob
        return len(pi_probs) - 1

    def forward_explore(self, train_step):
        terminal = False
        t_start = train_step
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        while not terminal and (train_step - t_start <= flags.t_max):
            pi_probs = self.local_net.get_policy(self.master.sess, self.env.state)
            if random.random() < 0.8:
                action = self.weighted_choose_action(pi_probs)
            else:
                action = random.randint(0, self.env.action_dim - 1)
            _, reward, terminal = self.env.forward_action(action)
            train_step += 1
            rollout_path["state"].append(self.env.state)
            one_hot_action = np.zeros(self.env.action_dim)
            one_hot_action[action] = 1
            rollout_path["action"].append(one_hot_action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal)
        return train_step, rollout_path

    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -flags.gamma], x[::-1], axis=0)[::-1]

    def run(self):
        sess = self.master.sess
        self.env.reset_env()
        loop = 0
        while flags.train_step <= flags.t_train:
            train_step = 0
            loop += 1
            # reset gradients
            sess.run(self.reset_accum_grads_ops)
            # sync variables
            sess.run(self.sync)
            # forward explore
            train_step, rollout_path = self.forward_explore(train_step)
            # rollout for discounted R values
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset_env()
                if flags.use_lstm:
                    self.local_net.reset_lstm_state()
            else:
                rollout_path["rewards"][-1] = self.local_net.get_value(sess, rollout_path["state"][-1])
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            # accumulate gradients
            lc_net = self.local_net
            fetches = [self.do_accum_grads_ops, self.master.global_step]
            if loop % 10 == 0:
                fetches.append(self.summary_op)
            res = sess.run(fetches, feed_dict={lc_net.state: rollout_path["state"],
                                               lc_net.action: rollout_path["action"],
                                               lc_net.target_q: rollout_path["returns"]})
            if loop % 10 == 0:
                global_step, summary_str = res[1:3]
                self.master.summary_writer.add_summary(summary_str, global_step=global_step)
                # logger.info("game=%s, global_step=%d, total_loss=%.2f" % (flags.game, global_step, total_loss))
                self.master.global_step_val = int(global_step)
            # async update grads to global network
            sess.run(self.apply_gradients)
            flags.train_step += train_step
            # evaluate
            if loop % 10 == 0 and self.thread_id == 1:
                self.test_phase()
            if loop % 1000 and self.thread_id == 1:
                save_model(self.master.sess, flags.train_dir, self.master.saver, "a3c_model",
                           global_step=self.master.global_step_val)

    def test_phase(self, episode=10, max_step=1e3):
        rewards = []
        start_time = time.time()
        while episode > 0:
            terminal = False
            self.env.reset_env()
            episode_reward = 0
            test_step = 0
            while not terminal and test_step < max_step:
                pi_probs = self.local_net.get_policy(self.master.sess, self.env.state)
                action = self.weighted_choose_action(pi_probs)
                _, reward, terminal = self.env.forward_action(action)
                test_step += 1
                episode_reward += reward
            rewards.append(episode_reward)
            episode -= 1
        elapsed_time = int(time.time() - start_time)
        avg_reward = float(np.mean(rewards))
        mid_reward = float(np.median(rewards))
        std_reward = float(np.std(rewards))
        logger.info("game=%s, train_step=%d, episode=%d, reward(avg:%.2f, mid:%.2f, std:%.2f), time=%d(s)" % (
            flags.game, flags.train_step, len(rewards), avg_reward, mid_reward, std_reward, elapsed_time))


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
    model.train()

if __name__ == "__main__":
    tf.app.run()
