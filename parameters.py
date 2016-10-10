import tensorflow as tf

tf.app.flags.DEFINE_string("game", "Breakout-v0", "gym environment name")
tf.app.flags.DEFINE_string("train_dir", "./models/experiment0/", "gym environment name")
tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
tf.app.flags.DEFINE_bool("use_lstm", False, "use LSTM layer")

tf.app.flags.DEFINE_integer("t_max", 600, "episode max time step")
tf.app.flags.DEFINE_integer("t_train", 1e9, "train max time step")
tf.app.flags.DEFINE_integer("jobs", 1, "parallel running thread number")

tf.app.flags.DEFINE_integer("frame_skip", 1, "number of frame skip")
tf.app.flags.DEFINE_integer("frame_seq", 4, "number of frame sequence")

tf.app.flags.DEFINE_string("opt", "rms", "choice in [rms, adam, sgd]")
tf.app.flags.DEFINE_float("learn_rate", 7e-4, "param of smooth")
tf.app.flags.DEFINE_integer("grad_clip", 40.0, "gradient clipping cut-off")
tf.app.flags.DEFINE_float("eps", 1e-8, "param of smooth")
tf.app.flags.DEFINE_float("entropy_beta", 1e-2, "param of policy entropy weight")
tf.app.flags.DEFINE_float("gamma", 0.95, "discounted ratio")
tf.app.flags.DEFINE_float("eGreedy", 0.8, "eGreedy")

tf.app.flags.DEFINE_bool("test", True, "test model")

tf.app.flags.DEFINE_float("train_step", 0, "train step. unchanged")

flags = tf.app.flags.FLAGS