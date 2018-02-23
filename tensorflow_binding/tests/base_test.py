import tensorflow as tf
from transducer_tensorflow import transducer_loss
targets = tf.sparse_placeholder(tf.int32)
trans_inputs = tf.placeholder(tf.float32, [None, 20, None])
predict_inputs= tf.placeholder(tf.float32, [None, 10, None])
input_lens = tf.placeholder(tf.int32, [20])
loss = transducer_loss(trans_inputs,predict_inputs,targets,input_lens)
print(loss)


