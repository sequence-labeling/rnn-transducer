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

        activations: A 3-D Tensor of floats.  The dimensions
                     should be (t, n, a), where t is the time index, n
                     is the minibatch index, and a indexes over
                     activations for each symbol in the alphabet.

        flat_labels: A 1-D Tensor of ints, a concatenation of all the
                     labels for the minibatch.
        input_lengths: A 1-D Tensor of ints, the number of time steps
                       for each sequence in the minibatch.

        blank_label: int, the label value/index that the CTC
                     calculation should use as the blank label

    Returns:
        1-D float Tensor, the cost of each example in the minibatch
        (as negative log probabilities).

    * This class performs the softmax operation internally.

    * The label reserved for the blank symbol should be label 0.

    '''
    if not isinstance(labels, sparse_tensor.SparseTensor):
       raise TypeError("Expected labels (first argument) to be a SparseTensor")
    loss, _,_ = _transducer.rnn_transducer(trans_acts,predict_acts,labels.indices,labels.values,input_lengths)
    return loss


@ops.RegisterGradient("RnnTransducer")
def _TransducerLossGrad(op, grad_losses, _):
    trans_grads = op.outputs[1]
    prdict_grads= op.outputs[2]
    return [_BroadcastMul(grad_losses[0], trans_grads), _BroadcastMul(grad_losses[1], predict_grads), None, None]


@ops.RegisterShape("RnnTransducer")
def _TransducerLossShape(op):
    trans_inputs_shape = op.inputs[0].get_shape().with_rank(3)
    predict_inputs_shape= op.inputs[1].get_shape().with_rank(3)
    batch_size = trans_inputs_shape[1]
    return [batch_size, trans_inputs_shape,predict_inputs_shape]

