import tensorflow as tf


lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

state = lstm.zero_state(batch_size, tf.float32)

loss = 0.0

for i in range(num_steps):
    if i > 0: tf.get_variable_scope().resue_variables()

    lstm_output, state = lstm(current_input, state)

    final_output = fully_connection(lstm_output)

    loss += calc_loss(final_output, expect_output)

