#include <tuple>
#include <limits>
#include <numeric>
#include <transducer_helper.h>
template<typename ProbT>
class CpuTransducer
{
public:
    CpuTransducer(int alphabet_size, int minibatch, void* workspace, int num_threads,
           int blank_label) :
            alphabet_size_(alphabet_size), minibatch_(minibatch),
            num_threads_(num_threads), workspace_(workspace),
            null_label_(null_label);
    transducerStatus_t cost_and_grad(const ProbT* const predict_act, const ProbT* const trans_act,
                              ProbT *predict_grads,ProbT *trans_grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    transducerStatus_t score_forward(const ProbT* const predict_act, const ProbT* const trans_act,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    private:
    int alphabet_size_; // Number of characters plus null label
    int minibatch_;
    int num_threads_;
    int null_label_;
    void* workspace_;
    void exp_matrix(Prob* predict_probs,Prob* trans_probs,const int* const input_lengths,const int* const label_lengths);
    ProbT compute_alphas(const ProbT* probs, int repeats, int S, int T,
                         const int* const e_inc,
                         const int* const s_inc,
                         const int* const labels,
                         ProbT* alphas);
     ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                 ProbT log_partition, int repeats,
                                 int S, int T, const int* const e_inc,
                                 const int* const s_inc,
                                 const int* const labels,
                                 ProbT* alphas,
                                 ProbT* betas,
                                 ProbT* output);

}
template<typename ProbT>
void
CpuTransducer<Prob>::exp_matrix(Prob* predict_probs,Prob* trans_probs,const int* const input_lengths,const int* const label_lengths)
{
 for (int mb = 0; mb < minibatch_; ++mb) {
      for(int c=0;c<input_lenths[mb];++c)
        {
            for(int r = 0; r < alphabet_size_; ++r)
             {
                int col_offset = (mb + minibatch_ * c) * alphabet_size_;
                trans_probs[r + col_offset] = std::exp(trans_probs[r+col_offset]);
                if(c<label_lengths[mb])
                {
                predict_probs[r + col_offset] = std::exp(predict_probs[r+col_offset]);
                }
              }
          }

}
}
template<typename ProbT>
void
CpuTransducer<ProbT>::softmax(const ProbT* const act1,const ProbT* const act2,ProbT* probs,const int* const input_lengths,const int* const label_lengths)
{
for(int mb=0;mb<minibatch_;++mb)
{
for
}
}
template<typename ProbT>
ProbT CpuTransducer<Prob>::compute_alphas()
{
//to do
}
template<typename ProbT>
tansducer<ProbT>::cost_and_grad(
const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths
)
{
if (activations == nullptr ||
        grads == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return TRANSDUCER_STATUS_INVALID_VALUE;
}
template<typename ProbT>
ProbT CpuTransducer<Prob>::compute_alphas(const ProbT* probs,int U,int T,ProbT* alphas)
{
for(int t=1;t<T;t++)
{
for(int u=0;u<U;u++)
{

}
}
} 
template<typename ProbT>
transducerStatus_t CpuTransducer<ProbT>::score_forward(const ProbT* const predict_act, const ProbT* const trans_act,
                                         ProbT* costs,
                                         const int* const flat_labels,
                                         const int* const label_lengths,
                                         const int* const input_lengths) {
    if (activations == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return TRANSDUCER_STATUS_INVALID_VALUE;
}}
