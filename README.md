# TensorFlow LSTMBlockCell with Stateful Inference
TensorFlow LSTMBlockCell with stateful mechanism for inference.

- Uses the improved LSTM implementation which is much faster: `tf.contrib.rnn.LSTMBlockCell`.
- Mechanism for saving and feeding LSTM states for 1-step inference.
- Classification task: predicting the number of cycles in a 1D signal (see the data generator).

My training runs achieve ~99.3% accuracy on the test set.

## References

- https://github.com/tensorflow/tensorflow/pull/2002
- https://github.com/tensorflow/tensorflow/issues/2695
