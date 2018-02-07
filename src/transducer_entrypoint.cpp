#include<transducer.h>
transducerStatus_t compute_ctc_loss(const float* const predict_probs,const float * const trans_probs,const int * const flat_labels,const int * const label_lengths,const int * const input lengths,int alphabet_size,int minibatch,float *costs,void *wordspace,transducerOptions options)
{
    if (trans_probs==nullptr||predict_probs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;
    if(options.loc==TRANSDUCER_CPU)
    {
      CpuTransducer<float> transducer(alphabet_size, minibatch, workspace, options.num_threads,options.null_label);
      if(gradients!=NULL)
          return transducer.cost_and_grad();
      else
          return transducer.score_forward();
    }
}
transducerStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcOptions options,
                               size_t* size_bytes)
{
  if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return TRANSDUCER_STATUS_INVALID_VALUE;
  int maxU=*std::max_element(label_lengths,label_lengths+minibatch);
  int maxT=*std::max_element(input_lengths,input_lengths+minibatch);
  if(options.loc==TRANSDUCER_CPU)
  {
      size_t per_minibatch_bytes=0;
      per_mini_batch_bytes+=sizeof(float)*alphabet_size;
      //the space for alphas
      per_minibatch_bytes+=sizeof(float)*maxU*maxT;
      //the space for betas
      per_minibatch_bytes+=sizeof(float)*maxU;
      //the space for probs
      per_minibatch_bytes+=sizeof(float)*alphabet_size * (maxT+maxU)
  }
  return TRANSDUCER_STATUS_SUCCESS;
//to do
}
