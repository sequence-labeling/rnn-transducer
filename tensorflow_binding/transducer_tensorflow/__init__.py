import imp
import tensorflow as tf
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

lib_file = imp.find_module('kernels', __path__)[1]
_transducer = tf.load_op_library(lib_file)

def transducer_loss(trans_acts,predict_acts,labels, input_lengths):
    '''Computes the CTC loss between a sequence of activations and a
    ground truth labeling.

    Args:


        trans_acts:3-D `float` `Tensor`.
              The transcription network output. shape must be [max_time,batch_size,num_classes,]
        predict_acts:3-D `float` `Tensor`.
              The predict network output.shape must be [max_label_lengths,batch_size,num_classes]
        labels: An `int32` `SparseTensor`.
             `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
              the id for (batch b, time t).
              `labels.values[i]` must take on values in `[0, num_labels)`.
              See `core/ops/ctc_ops.cc` for more details.
        input_lens: 1-D `int32` vector, size `[batch_size]`.
              The transcription network ouput lengths.

    Returns:
        1-D float Tensor, the cost of each example in the minibatch
    '''
    if not isinstance(labels, sparse_tensor.SparseTensor):
       raise TypeError("Expected labels (first argument) to be a SparseTensor")
    loss, _,_ = _transducer.rnn_transducer(trans_acts,predict_acts,labels.indices,labels.values,input_lengths)
    return loss


@ops.RegisterGradient("RnnTransducer")
def _TransducerLossGrad(op, grad_loss,grad1,grad2):
    trans_grads = op.outputs[1]
    predict_grads= op.outputs[2]
    return [_BroadcastMul(grad_loss, trans_grads), _BroadcastMul(grad_loss, predict_grads),None,None, None]


@ops.RegisterShape("RnnTransducer")
def _TransducerLossShape(op):
    trans_inputs_shape = op.inputs[0].get_shape().with_rank(3)
    predict_inputs_shape= op.inputs[1].get_shape().with_rank(3)
    batch_size = trans_inputs_shape[1]
    return [batch_size, trans_inputs_shape,predict_inputs_shape]

