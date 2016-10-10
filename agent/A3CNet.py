import tensorflow as tf
from common import *
from parameters import *

class A3CNet(object):
    def __init__(self, state_shape, action_dim, scope):
        with tf.device("/gpu:%d" % flags.gpu):
            # placeholder
            with tf.variable_scope("%s_holder" % scope):
                self.state = tf.placeholder(tf.float32, shape=[None] + list(state_shape), name="state")  # (None, 84, 84, 4)
                self.action = tf.placeholder(tf.float32, shape=[None, action_dim], name="action")  # (None, actions)
                self.target_q = tf.placeholder(tf.float32, shape=[None])
            # shared parts
            with tf.variable_scope("%s_shared" % scope):
                conv1, self.w1, self.b1 = conv2d(self.state, (8, 8, state_shape[-1], 16), "conv_1", stride=4,
                                                 padding="VALID", with_param=True, weight_decay=0.01, collect=scope)  # (None, 20, 20, 16)
                # conv1 = NetTools.batch_normalized(conv1)
                conv2, self.w2, self.b2 = conv2d(conv1, (4, 4, 16, 32), "conv_2", stride=2,
                                                 padding="VALID", with_param=True, weight_decay=0.01, collect=scope)  # (None, 9, 9, 32)
                # conv2 = NetTools.batch_normalized(conv2)
                flat1 = tf.reshape(conv2, (-1, 9 * 9 * 32), name="flat1")
                fc_1, self.w3, self.b3 = full_connect(flat1, (9 * 9 * 32, 256), "fc1", with_param=True, weight_decay=0.01, collect=scope)
                # fc_1 = NetTools.batch_normalized(fc_1)
            # policy parts
            with tf.variable_scope("%s_policy" % scope):
                pi_fc_1, self.pi_w1, self.pi_b1 = full_connect(fc_1, (256, 256), "pi_fc1", with_param=True, weight_decay=0.01, collect=scope)
                pi_fc_2, self.pi_w2, self.pi_b2 = full_connect(pi_fc_1, (256, action_dim), "pi_fc2", activate=None,
                                                               with_param=True)
                self.policy_out = tf.nn.softmax(pi_fc_2, name="pi_out")
            # value parts
            with tf.variable_scope("%s_value" % scope):
                v_fc_1, self.v_w1, self.v_b1 = full_connect(fc_1, (256, 256), "v_fc1", with_param=True, weight_decay=0.01, collect=scope)
                v_fc_2, self.v_w2, self.v_b2 = full_connect(v_fc_1, (256, 1), "v_fc2", activate=None, with_param=True)
                self.value_out = tf.reshape(v_fc_2, [-1], name="v_out")
            # loss values
            with tf.op_scope([self.policy_out, self.value_out], "%s_loss" % scope):
                self.entropy = - tf.reduce_sum(self.policy_out * tf.log(self.policy_out + flags.eps))
                time_diff = self.target_q - self.value_out
                policy_prob = tf.log(tf.reduce_sum(tf.mul(self.policy_out, self.action), reduction_indices=1))
                self.policy_loss = - tf.reduce_sum(policy_prob * time_diff)
                self.value_loss = tf.reduce_sum(tf.square(time_diff))
                self.l2_loss = tf.add_n(tf.get_collection(scope))
                self.total_loss = self.policy_loss + self.value_loss * 0.5 + self.entropy * flags.entropy_beta

    def get_policy(self, sess, state):
        return sess.run(self.policy_out, feed_dict={self.state: [state]})[0]

    def get_value(self, sess, state):
        return sess.run(self.value_out, feed_dict={self.state: [state]})[0]

    def get_vars(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                self.pi_w1, self.pi_b1, self.pi_w2, self.pi_b2,
                self.v_w1, self.v_b1, self.v_w2, self.v_b2]
