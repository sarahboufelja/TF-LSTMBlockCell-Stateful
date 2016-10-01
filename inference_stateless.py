import numpy as np
import tensorflow as tf


################################################################################

test_data   = np.load("./data/tst_inputs.npy")
test_labels = np.load("./data/tst_labels.npy")

num_examples = test_labels.shape[0]
num_labels   = test_labels.shape[1]

seq_length   = 64
batch_size   = 1
data_dim     = 1

lstm_units = 256

################################################################################
############################ MODEL DEFINITION ##################################
################################################################################

x = tf.placeholder(tf.float32, [None, seq_length, data_dim])
y = tf.placeholder(tf.float32, [None, num_labels])
dropout_keep_prob = tf.placeholder(tf.float32, [])

# Note: using the LSTMBlockCell
rnn_cell = tf.contrib.rnn.LSTMBlockCell(lstm_units, forget_bias=1.0)
rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep_prob)
rnn = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * 1)  # single layer
rnn_state = rnn.zero_state(batch_size, tf.float32)

weight_init = tf.contrib.layers.variance_scaling_initializer()
output_W = tf.get_variable("W", shape=[lstm_units, num_labels], initializer=weight_init)
output_b = tf.get_variable("b", shape=[num_labels], initializer=tf.constant_initializer(0.0))

# Split into timesteps
x_split = tf.split(1, seq_length, x)

# LSTM unrolling for timesteps
for step in range(seq_length):
    with tf.variable_scope("RNN") as scope:
        if step > 0: scope.reuse_variables()
        input_step = tf.squeeze(x_split[step], [1])
        h_rnn, rnn_state = rnn(input_step, rnn_state)

# Outputs
logits = tf.nn.bias_add(tf.matmul(h_rnn, output_W), output_b, name="logits")
probabilities = tf.nn.softmax(logits)
predictions = tf.argmax(probabilities, dimension=1)

loss_per_example = tf.nn.softmax_cross_entropy_with_logits(logits, y)
loss = tf.reduce_mean(loss_per_example)

# Prediction accuracy
accurcy = tf.contrib.metrics.accuracy(predictions, tf.argmax(y, dimension=1))

################################################################################

# Create a saver for all variables
tf_vars_to_save = tf.trainable_variables()
saver = tf.train.Saver(tf_vars_to_save)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Loading checkpoint file from disk
latest_checkpoint = tf.train.latest_checkpoint("./checkpoints/")
saver.restore(sess, latest_checkpoint)

test_losses     = np.zeros(num_examples, dtype=np.float32)
test_accuracies = np.zeros(num_examples, dtype=np.float32)

for i in range(num_examples):

    feed_dict = {
        x: np.expand_dims(test_data[i,], 0),
        y: np.expand_dims(test_labels[i,], 0),
        dropout_keep_prob: 1.0
    }

    test_loss, test_acc = sess.run([loss, accurcy], feed_dict=feed_dict)

    if i % 100 == 0:
        print("Example %03i, Test Accuracy = %.2f, Test Loss = %.3f" % (i, test_acc, test_loss))

    # Save average accuracies and losses per batch
    test_losses[i]     = test_loss
    test_accuracies[i] = test_acc

print("Model performance on test set:")
print("  Average Test Accuracy: %.3f" % np.mean(test_accuracies))
print("  Average Test Loss:     %.3f" % np.mean(test_losses))


