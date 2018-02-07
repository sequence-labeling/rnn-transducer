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

//to do
}
