import numpy as np
import tensorflow as tf


################################################################################

trn_data   = np.load("./data/trn_inputs.npy")
trn_labels = np.load("./data/trn_labels.npy")
val_data   = np.load("./data/val_inputs.npy")
val_labels = np.load("./data/val_labels.npy")

num_trn_examples = trn_labels.shape[0]
num_val_examples = val_labels.shape[0]
num_labels = trn_labels.shape[1]

train_steps   = 20000
seq_length    = 64
batch_size    = 128
data_dim      = 1
learning_rate = 0.01
dropout_kp    = 0.5
num_epochs    = 50
lstm_units    = 256

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

# Compute batch loss
loss_per_example = tf.nn.softmax_cross_entropy_with_logits(logits, y)
loss = tf.reduce_mean(loss_per_example)

# Prediction accuracy
accurcy = tf.contrib.metrics.accuracy(predictions, tf.argmax(y, dimension=1))

################################################################################
################################ TRAIN OPS #####################################
################################################################################

global_step = tf.Variable(0, trainable=False)

optimizer = tf.train.RMSPropOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)

# Gradient clipping
grads, variables = zip(*grads_and_vars)
grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
grads_and_vars = zip(grads_clipped, variables)

train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Create a saver for all variables
tf_vars_to_save = tf.trainable_variables() + [global_step]
saver = tf.train.Saver(tf_vars_to_save, max_to_keep=5)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

################################################################################
################################ TRAIN LOOP ####################################
################################################################################

index_in_epoch    = 0
batches_per_epoch = int(np.floor_divide(num_trn_examples, batch_size))


for _ in range(train_steps):

    start = index_in_epoch*batch_size
    end   = start+batch_size

    feed_dict = {
        x: trn_data[start:end,],
        y: trn_labels[start:end,],
        dropout_keep_prob: dropout_kp
    }

    _, step, train_loss, train_acc = sess.run([train_op, global_step, loss, accurcy], feed_dict=feed_dict)
    index_in_epoch += 1

    if step % 20 == 0:
        print("Step %05i, Train Accuracy = %.2f, Train Loss = %.3f" % (step, train_acc, train_loss))

    if step % batches_per_epoch == 0:

        # Check performance on validations set
        print("Testing model performance on validation set:")
        num_val_batches = int(np.floor_divide(num_trn_examples, float(batch_size)))
        val_losses     = np.zeros(num_val_batches, dtype=np.float32)
        val_accuracies = np.zeros(num_val_batches, dtype=np.float32)

        for i in range(num_val_batches):

            start = i * batch_size
            end = start + batch_size

            feed_dict = {
                x: val_data[start:end, ],
                y: val_labels[start:end, ],
                dropout_keep_prob: 1.0
            }

            val_loss, val_acc = sess.run([loss, accurcy], feed_dict=feed_dict)
            # print("  %03i. Validation Accuracy = %.2f, Validation Loss = %.3f" % (i, val_acc, val_loss))

            val_losses[i]     = val_loss
            val_accuracies[i] = val_acc

        print("  Average Validation Accuracy: %.3f" % np.mean(val_accuracies))
        print("  Average Validation Loss:     %.3f" % np.mean(val_losses))

        # Shuffling training data for next epoch
        perm = np.arange(len(trn_data))
        np.random.shuffle(perm)
        trn_data = trn_data[perm]
        trn_labels = trn_labels[perm]
        index_in_epoch = 0

        # Save the checkpoint to disk
        path = saver.save(sess, "./checkpoints/chkpt", global_step=global_step)
        print("Checkpoint saved to %s" % path)

print("Training done.")


