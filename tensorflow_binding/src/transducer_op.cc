#ifdef TRANSDUCER_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cuda.h>
#endif
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"
#include "transducer.h"
REGISTER_OP("transducer")
      .Input("trans_acts: float32")
      .Input("predict_acts: float32")
      .Input("input_lengths: int32")
      .Input("label_lengths: int32")
      .Input("null_label:int =0")
      .Output("costs: float32")
      .Output("grads_trans: float32")
      .Output("grads_predict:float32 ");
namespace tf=tensorflow
namespace transducer{
class TransducerOpBase:public tf::OpKernel
        {
            public:
                explicit TransducerOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("null_label", &null_label_));
        }

        void Compute(tf::OpKernelContext* ctx) override
        {
        const tf::Tensor * trans_acts;
        const tf::Tensor * predict_act;
        const tf::Tensor * input_lengths;
        const tf::Tensor * label_lengths;
        cons
        
        }

}
