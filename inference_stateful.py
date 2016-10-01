import numpy as np
import tensorflow as tf


################################################################################

test_data   = np.load("./data/tst_inputs.npy")
test_labels = np.load("./data/tst_labels.npy")

num_examples = test_labels.shape[0]
num_labels   = test_labels.shape[1]

seq_length   = 1
batch_size   = 1
data_dim     = 1

lstm_units = 256

################################################################################

# https://github.com/tensorflow/tensorflow/issues/2695#issuecomment-237097094
def rnn_states_list(states, state_is_tuple):
    """
    given a 'states' variable from a tensorflow model,
    return a flattened list of states
    """
    if state_is_tuple:
        states_list = [] # flattened list of all tensors in states
        for layer in states:
            for state in layer:
                states_list.append(state)
    else:
        states_list = [states]

    return states_list


def rnn_states_dict(sess, states, state_is_tuple):
    """
    given a 'states' variable from a tensorflow model,
    return a dict of { tensor : evaluated value }
    """
    if state_is_tuple:
        states_dict = {} # dict of { tensor : value }
        for layer in states:
            for state in layer:
                states_dict[state] = sess.run(state)

    else:
        states_dict = {states : sess.run(states)}

    return states_dict


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
rnn_init_state = rnn.zero_state(batch_size, tf.float32)

weight_init = tf.contrib.layers.variance_scaling_initializer()
output_W = tf.get_variable("W", shape=[lstm_units, num_labels], initializer=weight_init)
output_b = tf.get_variable("b", shape=[num_labels], initializer=tf.constant_initializer(0.0))

# Split into timesteps
x_split = tf.split(1, seq_length, x)

# Set initial LSTM state
rnn_state = rnn_init_state

# NOTE: only one timestep, leave code similar to train/stateless for readibility
for step in range(seq_length):
    with tf.variable_scope("RNN") as scope:
        if step > 0: scope.reuse_variables()
        input_step = tf.squeeze(x_split[step], [1])
        h_rnn, rnn_state = rnn(input_step, rnn_state)

# Keep track of the final RNN state
rnn_final_state = rnn_state

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

# Initialial and final LSTM states for stateful model
rnn_init_states_list  = rnn_states_list(rnn_init_state, True)
rnn_init_states_dict  = rnn_states_dict(sess, rnn_init_state, True)
rnn_final_states_list = rnn_states_list(rnn_final_state, True)

# Loading checkpoint file from disk
latest_checkpoint = tf.train.latest_checkpoint("./checkpoints/")
saver.restore(sess, latest_checkpoint)

test_losses     = np.zeros(num_examples, dtype=np.float32)
test_accuracies = np.zeros(num_examples, dtype=np.float32)

# Different from seq_length, which is set to 1 for stateful inference
timesteps = test_data.shape[1]

for i in range(num_examples):

    test_loss = 0.0
    test_acc  = 0.0

    # Label is the same for each timestep, actually we do not really need it for every step.
    example_label = test_labels[i,]

    # Set initial hidden state (zeros)
    states_dict = rnn_init_states_dict

    for step in range(timesteps):

        input_step = np.expand_dims(np.expand_dims(test_data[i,step,], 0), 0)

        feed_dict = {
            x: input_step, # <== input for only one timestep
            y: np.expand_dims(example_label, 0),
            dropout_keep_prob: 1.0,
        }

        # Set initial RNN states
        feed_dict.update(states_dict)

        # Fetch the loss, accuracy and the LSTM states after this timestep
        fetch = [loss, accurcy] + rnn_final_states_list

        # Also fetch the final LSTM states. Note that the loss and accuracy do not make sense except for the last step.
        output = sess.run(fetch, feed_dict=feed_dict)

        # Process the model outputs
        test_loss = output[0]
        test_acc  = output[1]
        states    = output[2:]
        states_dict = dict(zip(rnn_init_states_list, states))

    if i % 100 == 0:
        print("Example %03i, Test Accuracy = %.2f, Test Loss = %.3f" % (i, test_acc, test_loss))

    # Save average accuracies and losses per batch
    test_losses[i]     = test_loss
    test_accuracies[i] = test_acc

print("Model performance on test set:")
print("  Average Test Accuracy: %.3f" % np.mean(test_accuracies))
print("  Average Test Loss:     %.3f" % np.mean(test_losses))


