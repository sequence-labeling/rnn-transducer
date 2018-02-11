#include<transducer.h>
#include <detail/cpu_transducer.h>
#include<iostream>
#include<cstddef>
int get_transducer_version() {
    return 1;
}
const char* transducerGetStatusString(transducerStatus_t status) {
    switch (status) {
    case TRANSDUCER_STATUS_SUCCESS:
        return "no error";
    case TRANSDUCER_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case TRANSDUCER_STATUS_INVALID_VALUE:
        return "invalid value";
    case TRANSDUCER_STATUS_EXECUTION_FAILED:
        return "execution failed";

    case TRANSDUCER_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";

    }

}
transducerStatus_t compute_transducer_loss(const float* const predict_acts,const float * const trans_acts,float * trans_grads,float * predict_grads, const int const *flat_labels,const int * const label_lengths,const int * const input_lengths,int alphabet_size,int minibatch,float *costs,void *workspace,transducerOptions options)
{
    if (trans_acts==nullptr||predict_acts == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return TRANSDUCER_STATUS_INVALID_VALUE;
    if(options.loc==TRANSDUCER_CPU)
    {
      CpuTransducer<float> transducer(alphabet_size, minibatch, workspace, options.num_threads,options.null_label);
      /*if(trans_grads!=NULL&&predict_grads!=NULL)
          return transducer.cost_and_grad();
      else*/
//          std::cout<<"enter transducer cpu";
          return transducer.score_forward(predict_acts,trans_acts,costs,flat_labels,label_lengths,input_lengths);
    }
}
transducerStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               transducerOptions options,
                               size_t* size_bytes)
{
  if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return TRANSDUCER_STATUS_INVALID_VALUE;
  int maxU=*std::max_element(label_lengths,label_lengths+minibatch)+1;
  int maxT=*std::max_element(input_lengths,input_lengths+minibatch);
  if(options.loc==TRANSDUCER_CPU)
  {
      size_t per_minibatch_bytes=0;
      //per_minibatch_bytes+=sizeof(float)*alphabet_size;
      //the space for alphas
      per_minibatch_bytes+=sizeof(float)*maxU*maxT;
      //the space for betas
      //per_minibatch_bytes+=sizeof(float)*maxU;
      //the space for probs
      per_minibatch_bytes+=sizeof(float)*alphabet_size * (maxT+maxU);
      per_minibatch_bytes+=sizeof(float)*alphabet_size* maxT*maxU;
      *size_bytes=per_minibatch_bytes*minibatch;
  }
  return TRANSDUCER_STATUS_SUCCESS;
//to do
}
