from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from parameters import *
from common import *

class A3CLSTMNet(object):
    def __init__(self, state_shape, action_dim, scope):

        class InnerLSTMCell(BasicLSTMCell):
            def __init__(self, num_units, forget_bias=1.0, input_size=None):
                BasicLSTMCell.__init__(self, num_units, forget_bias=forget_bias, input_size=input_size)
                self.matrix, self.bias = None, None

            def __call__(self, inputs, state, scope=None):
                """
                    Long short-term memory cell (LSTM).
                    implement from BasicLSTMCell.__call__
                """
                with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
                    # Parameters of gates are concatenated into one multiply for efficiency.
                    c, h = tf.split(1, 2, state)
                    concat = self.linear([inputs, h], 4 * self._num_units, True)

                    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                    i, j, f, o = tf.split(1, 4, concat)

                    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
                    new_h = tf.tanh(new_c) * tf.sigmoid(o)

                    return new_h, tf.concat(1, [new_c, new_h])

            def linear(self, args, output_size, bias, bias_start=0.0, scope=None):
                """
                    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
                    implement from function of tensorflow.python.ops.rnn_cell.linear()
                """
                if args is None or (isinstance(args, (list, tuple)) and not args):
                    raise ValueError("`args` must be specified")
                if not isinstance(args, (list, tuple)):
                    args = [args]

                    # Calculate the total size of arguments on dimension 1.
                total_arg_size = 0
                shapes = [a.get_shape().as_list() for a in args]
                for shape in shapes:
                    if len(shape) != 2:
                        raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
                    if not shape[1]:
                        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
                    else:
                        total_arg_size += shape[1]

                # Now the computation.
                with tf.variable_scope(scope or "Linear"):
                    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
                    if len(args) == 1:
                        res = tf.matmul(args[0], matrix)
                    else:
                        res = tf.matmul(tf.concat(1, args), matrix)
                    if not bias:
                        return res
                    bias_term = tf.get_variable(
                        "Bias", [output_size],
                        initializer=tf.constant_initializer(bias_start))
                    self.matrix = matrix
                    self.bias = bias_term
                return res + bias_term

        with tf.device("/gpu:%d" % flags.gpu):
            # placeholder
            self.state = tf.placeholder(tf.float32, shape=[None] + list(state_shape), name="state")  # (None, 84, 84, 4)
            self.action = tf.placeholder(tf.float32, shape=[None, action_dim], name="action")  # (None, actions)
            self.target_q = tf.placeholder(tf.float32, shape=[None])
            # shared parts
            with tf.variable_scope("%s_shared" % scope):
                conv1, self.w1, self.b1 = conv2d(self.state, (8, 8, state_shape[-1], 16), "conv_1", stride=4,
                                                 padding="VALID", with_param=True)  # (None, 20, 20, 16)
                conv2, self.w2, self.b2 = conv2d(conv1, (4, 4, 16, 32), "conv_2", stride=2,
                                                 padding="VALID", with_param=True)  # (None, 9, 9, 32)
                flat1 = tf.reshape(conv2, (9 * 9 * 32, 256), name="flat1")
                fc_1, self.w3, self.b3 = full_connect(flat1, (9 * 9 * 32, 256), "fc1", with_param=True)
            # rnn parts
            with tf.variable_scope("%s_rnn" % scope) as scope:
                h_flat1 = tf.reshape(fc_1, (1, -1, 256))
                self.lstm = InnerLSTMCell(256)
                self.initial_lstm_state = tf.placeholder(tf.float32, shape=[1, self.lstm.state_size])
                self.sequence_length = tf.placeholder(tf.float32, [1])
                lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm, h_flat1,
                                                                  initial_state=self.initial_lstm_state,
                                                                  sequence_length=self.sequence_length,
                                                                  time_major=False,
                                                                  scope=scope)
                lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
            # policy parts
            with tf.variable_scope("%s_policy" % scope):
                pi_fc_1, self.pi_w1, self.pi_b1 = full_connect(lstm_outputs, (256, 256), "pi_fc1", with_param=True)
                pi_fc_2, self.pi_w2, self.pi_b2 = full_connect(pi_fc_1, (256, action_dim), "pi_fc2", activate=None,
                                                               with_param=True)
                self.policy_out = tf.nn.softmax(pi_fc_2, name="pi_out")
            # value parts
            with tf.variable_scope("%s_value" % scope):
                v_fc_1, self.v_w1, self.v_b1 = full_connect(lstm_outputs, (256, 256), "v_fc1", with_param=True)
                v_fc_2, self.v_w2, self.v_b2 = full_connect(v_fc_1, (256, 1), "v_fc2", activate=None, with_param=True)
                self.value_out = tf.reshape(v_fc_2, [-1], name="v_out")
            # loss values
            with tf.op_scope([self.policy_out, self.value_out], "%s_loss" % scope):
                self.entropy = - tf.reduce_mean(self.policy_out * tf.log(self.policy_out + flags.eps))
                time_diff = self.target_q - self.value_out
                policy_prob = tf.log(tf.reduce_sum(tf.mul(self.policy_out, self.action), reduction_indices=1))
                self.policy_loss = - tf.reduce_sum(policy_prob * time_diff, reduction_indices=1)
                self.value_loss = tf.square(time_diff)
                self.total_loss = self.policy_loss + self.value_loss * 0.5 + self.entropy * flags.entropy_beta
        # lstm state
        self.lstm_state_out = np.zeros((1, self.lstm.state_size), dtype=np.float32)

    def reset_lstm_state(self):
        self.lstm_state_out = np.zeros((1, self.lstm.state_size), dtype=np.float32)

    def get_policy(self, sess, state):
        policy_out, self.lstm_state_out = sess.run([self.policy_out, self.lstm_state],
                                                   feed_dict={self.state: [state],
                                                              self.initial_lstm_state: self.lstm_state,
                                                              self.sequence_length: [1]})
        return policy_out[0]

    def get_value(self, sess, state):
        value_out, _ = sess.run([self.value_out, self.lstm_state], feed_dict={self.state: [state],
                                                                              self.initial_lstm_state: self.lstm_state,
                                                                              self.sequence_length: [1]})[0]
        return value_out[0]

    def get_vars(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                self.lstm.matrix, self.lstm.bias,
                self.pi_w1, self.pi_b1, self.pi_w2, self.pi_b2,
                self.v_w1, self.v_b1, self.v_w2, self.v_b2]