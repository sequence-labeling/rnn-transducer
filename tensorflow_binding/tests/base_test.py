import numpy as np
import tensorflow as tf
from transducer_tensorflow import transducer_loss
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
 
    return indices, values, shape
targets = tf.sparse_placeholder(tf.int32)
trans_inputs = tf.placeholder(tf.float32, [None, 1, None])
predict_inputs= tf.placeholder(tf.float32, [None, 1, None])
seq_lens = tf.placeholder(tf.int32, [1])
loss = transducer_loss(trans_inputs,predict_inputs,targets,seq_lens)
grad = tf.gradients(loss, [trans_inputs,predict_inputs])
trans_grad=grad[0]
predict_grad=grad[1]

print(loss.get_shape())

trans_act=np.array([[[0.1, 0.9, 0.1, 0.1, 0.1]],[[0.1, 0.1, 0.9, 0.1, 0.1]]],dtype=np.float32);
predict_act=np.array([[[0.1, 0.9, 0.1, 0.1, 0.1]],[[0.1, 0.1, 0.9, 0.1, 0.1]], [[2, 0.1, 0, 0.1, 0.5]]],dtype=np.float32);
input_lengths=np.asarray([2],dtype=np.int32)
target=[[0,1]];
test_dict= {trans_inputs: trans_act,predict_inputs:predict_act,targets:sparse_tuple_from(target),seq_lens:input_lengths}
with tf.Session() as session:
   cost,trans_grad,predict_grad=session.run([loss,trans_grad,predict_grad],feed_dict=test_dict)
   print(cost)
   print(trans_grad)
   print(predict_grad)

