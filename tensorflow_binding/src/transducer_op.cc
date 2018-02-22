#ifdef WARPCTC_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cuda.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "transducer.h"
REGISTER_OP("transducer")
    .Input("trans_acts: float32")
    .Input("predict_acts: float32")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("input_lengths: int32")
    .Output("costs: float32")
    .Output("trans_grads: float32")
    .Output("predict_grads :float32");
namespace tf = tensorflow;

namespace transducer_transducer {

class TransducerLossOpBase : public tf::OpKernel {
  public:
    explicit TransducerLossOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    }

    void Compute(tf::OpKernelContext* ctx) override {
        const tf::Tensor* trans_inputs;
        const tf::Tensor* predict_inputs;
        const tf::Tensor* labels_indices;
        const tf::Tensor* labels_values;
        const tf::Tensor* input_lens;
        OP_REQUIRES_OK(ctx, ctx->input("trans_acts", &trans_inputs));
        OP_REQUIRES_OK(ctx,ctx->input("predict_acts",&predict_inputs));
        OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
        OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
        OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lens));
        OP_REQUIRES(ctx, trans_inputs->shape().dims() == 3,
                    tf::errors::InvalidArgument("transcription network inputs is not a 3-Tensor"));
        OP_REQUIRES(ctx, predict_inputs->shape().dims() == 3,
                     tf::errors::InvalidArgument("predict network inputs is not a 3-Tensor"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lens->shape()),
                    tf::errors::InvalidArgument("input_lengths is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsMatrix(labels_indices->shape()),
                    tf::errors::InvalidArgument("labels_indices is not a matrix"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(labels_values->shape()),
                    tf::errors::InvalidArgument("labels_values is not a vector"));

        const auto& trans_shape = trans_inputs->shape();
        const auto& predict_shape=predict_inputs->shape();
        const auto max_time = trans_shape.dim_size(0);
        const auto batch_size = trans_shape.dim_size(1);
        const auto num_classes_raw = trans_shape.dim_size(2);
        OP_REQUIRES(
                ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
                tf::errors::InvalidArgument("num_classes cannot exceed max int"));
        const auto num_classes = static_cast<const int>(num_classes_raw);

        OP_REQUIRES(
                ctx, batch_size == input_lens->dim_size(0),
                tf::errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                            "len(sequence_length):  ", input_lens->dim_size(0),
                                            " batch_size: ", batch_size));
        auto input_lens_t = input_lens->vec<int32_t>();

        OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                    tf::errors::InvalidArgument(
                            "labels_indices and labels_values must contain the "
                            "same number of rows, but saw shapes: ",
                            labels_indices->shape().DebugString(), " vs. ",
                            labels_values->shape().DebugString()));

        auto labels_shape = tf::TensorShape({batch_size, max_time});
        auto order = std::vector<tf::int64>{0, 1};
        auto labels_sp = tf::sparse::SparseTensor(*labels_indices, *labels_values,
                                                  labels_shape, order);

        auto labels_sp_valid = labels_sp.IndicesValid();
        OP_REQUIRES(ctx, labels_sp_valid.ok(),
                    tf::errors::InvalidArgument("label SparseTensor is not valid: ",
                                            labels_sp_valid.error_message()));

        auto label_lengths = std::vector<int>{};
        for (const auto& g : labels_sp.group({0})) {  // iterate by batch
            const auto batch_indices = g.group()[0];
            OP_REQUIRES(ctx, tf::FastBoundsCheck(batch_indices, batch_size),
                        tf::errors::InvalidArgument("labels batch index must be between ",
                                                    0, " and ", batch_size, " but saw: ",
                                                    batch_indices));
            
            auto values = g.values<int32_t>();
            label_lengths.push_back(values.size());
        }
        auto label_values_t = labels_values->vec<int>();


        OP_REQUIRES(ctx, static_cast<size_t>(batch_size) == label_lengths.size(),
                    tf::errors::InvalidArgument("len(labels) != batch_size.  ",
                                                "len(labels):  ", label_lengths.size(),
                                                " batch_size: ", batch_size));

        for (int b = 0; b < batch_size; ++b) {
            OP_REQUIRES(
                    ctx, input_lens_t(b) <= max_time,
                    tf::errors::InvalidArgument("sequence_length(", b, ") <= ", max_time));
        }

        tf::Tensor* loss = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", input_lens->shape(), &loss));
        auto loss_t = loss->vec<float>();

        tf::Tensor* trans_grads;
        tf::Tensor* predict_grads;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output("trans_grads", trans_shape, &trans_grads));
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output("predict_grads", predict_shape, &predict_grads));
        set_zero(trans_grads);
        set_zero(predict_grads);
        auto trans_gradient_t = trans_grads->tensor<float, 3>();
        auto predict_gradient_t=predict_grads->tensor<float,3>();

        auto trans_inputs_t = trans_inputs->tensor<float, 3>();
        auto predict_inputs_t=predict_inputs->tensor<float,3>();
        auto options = create_options(ctx);
        options.null_label = num_classes - 1;

        size_t workspace_size_bytes;
        auto transducer_status = get_workspace_size(input_lens_t.data(),label_lengths.data(),
                                              num_classes, batch_size,
                                              options, &workspace_size_bytes);
        OP_REQUIRES(ctx, transducer_status == TRANSDUCER_STATUS_SUCCESS,
                    tf::errors::Internal("transducer error in get_workspace_size: ",
                                         transducerGetStatusString(transducer_status)));

        auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
        tf::Tensor workspace;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
        auto workspace_t = workspace.flat<uint8_t>();

        transducer_status = compute_transducer_loss(trans_inputs_t.data(),predict_inputs_t.data(),
                                       trans_gradient_t.data(),
                                       predict_gradient_t.data(),
                                       label_values_t.data(),
                                       input_lens_t.data(),
                                       label_lengths.data(),
                                       num_classes, batch_size,
                                       loss_t.data(), workspace_t.data(), options);
        
        OP_REQUIRES(ctx, transducer_status == TRANSDUCER_STATUS_SUCCESS,
                    tf::errors::Internal("transducer_ctc error in compute_transducer_loss: ",
                                         transducerGetStatusString(transducer_status)));

    }

  private:
    virtual void set_zero(tf::Tensor* t) = 0;
    virtual transducerOptions create_options(tf::OpKernelContext* ctx) = 0;
};

class TransducerLossOpCPU : public TransducerLossOpBase {
  public:
    explicit TransducerLossOpCPU(tf::OpKernelConstruction* ctx) : TransducerLossOpBase(ctx) {
    }

  private:
    void set_zero(tf::Tensor* t) override {
        t->flat<float>().setZero();
    }

    transducerOptions create_options(tf::OpKernelContext* ctx) override {
        auto options = transducerOptions{};
        options.loc = TRANSDUCER_CPU;
        options.num_threads = ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("TransducerLoss").Device(::tensorflow::DEVICE_CPU).Label("transducer"),TransducerLossOpCPU);
}
