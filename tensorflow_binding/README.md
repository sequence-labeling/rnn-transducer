#tensorFlow binding for rnn-transducer
This package provide TensorFlow kernel that warp rnn transducer
##Instation
To build the kernels it is necessary to have the TensorFlow source code availabe,since TensorFlow doesn't currently install the
necessary headers to handle the SparseTensor that the CTCLoss op uses
to input the labels.  You can retrieve the TensorFlow source from
github.com:
```bash
git clone https://github.com/tensorflow/tensorflow.git
```

Tell the build scripts where you have the TensorFlow source tree by
setting the `TENSORFLOW_SRC_PATH` environment variable:

```bash
export TENSORFLOW_SRC_PATH=/path/to/tensorflow
```

`RNN_TRANSDUCER_PATH` should be set to the location of a built rnn-transducer
(i.e. `libtransducer.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build rnn-transducer like this:

```bash
mkdir build; cd build
cmake ..
make
```
Otherwise, set `RNN_TRANSDUCER_PATH` to wherever you have `libtransducer.so`

You should now be able to use `setup.py` to install the package into
your current Python environment:

```bash
python setup.py install
```

## Using the kernels
```python
import transducer_tensorflow
```
```python
costs = transducer_tensorflow.transducer_loss(trans_inputs,predict_inputs,labels,seq_lens)

```

trans_inputs:3-D `float` `Tensor`.
      The transcription network output. shape must be [max_time,batch_size,num_classes,]
predict_inputs:3-D `float` `Tensor`.
      The predict network output.shape must be [max_label_lengths,batch_size,num_classes]
labels: An `int32` `SparseTensor`.
      `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
      the id for (batch b, time t).
      `labels.values[i]` must take on values in `[0, num_labels)`.
      See `core/ops/ctc_ops.cc` for more details.
seq_lens: 1-D `int32` vector, size `[batch_size]`.
      The transcription network output lengths.

