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
    class CpuTransducer_metadata
    {
     public  CpuCTC_metadata(int U, int T, int mb, int alphabet_size,
                        void* workspace, size_t bytes_used, int blank_label,
                        const int* const labels);
     ProbT* alphas;
     ProbT* betas;
     ProbT* output;
    }
    int alphabet_size_; // Number of characters plus null label
    int minibatch_;
    int num_threads_;
    int null_label_;
    unoredered_map<pair<int,pair<int,int>>,ProbT> probs_utk;
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
CpuTransducer<ProbT>::CpuTransducer_metadata::CpuTransducer_metadata(int U, int T, int mb,
                                                int alphabet_size,
                                                void* workspace, size_t bytes_used,
                                                int null_label,
                                                const int* const labels) {

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * U * T;
    std::fill(alphas, alphas + U* T,1);
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) *U;
    std::fill(betas, betas + S, 1);
    labels_w_blanks = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    e_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    s_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    output = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * alphabet_size;

    repeats = setup_labels(labels, blank_label, L, S);
}
CpuTransducer<Prob>::exp_matrix(const ProbT* const trans_act,const ProbT*  const predict_probs,ProbT* trans_exp,ProbT* predict_exp,const int* const input_lengths,const int* const label_lengths)
{
 for (int mb = 0; mb < minibatch_; ++mb) {
      for(int c=0;c<input_lenths[mb];++c)
        {
            for(int r = 0; r < alphabet_size_; ++r)
             {
                int col_offset = (mb + minibatch_ * c) * alphabet_size_;
                trans_exp[r + col_offset] = std::exp(trans_act[r+col_offset]);
                if(c<label_lengths[mb])
                {
                predict_exp[r + col_offset] = std::exp(predict_act[r+col_offset]);
                }
              }
          }

}
}
template<typename Prob>
void
CpuTransduer<ProbT>::compute_pr(const ProbT* const trans_exp,const ProbT* predict_exp,ProbT* const probs_utk,int maxU,int maxT,const int * const input_lengths,const int * const label_lengths,int *labels)
{
   ProbT sum=0,prob_null,prob_back,prob_forword;
   int null_index,forward_index,back_index,col_offset
   for(int mb=0;mb<minibatch_;++mb)
   {
     for(int t=0;t<input_lengths[mb];++t)
     {
       int trans_col_offset=(mb+minibatch_*t)*alphabet_size_;
       for(int u=0;u<label_lengths[mb];++u)
       {
         null_index=(mb+minibatch*t)*(maxU*alphabet_size)+u;
         forward_index=null_index+max_U;
         backward_index=forward_index+max_U;
         int null_index(mb+mi)
         int predict_col_offset=(mb+minibatch_*u)*alphabet_size;
         sum=0;
         for(int r=0;r<alphabet_size_;c++)
         {
           ProbT tmp=trans_act[trans_col_offset+r]*predict_act[predict_col_offset+r];
           probs_utk[null_index+r*alphabet_size]=tmp/sum;
         }
       }
     }
   }
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

//t,u   u*T+t
//u: [0,U]
//t: [0,T)
//must invoke compute_pr before invoke this function
template<typename ProbT>
ProbT CpuTransducer<Prob>::compute_alphas(const ProbT* const probs_ut,int max_U,int U,int * labels,int T)
{
    std::fill(alphas, alphas + (U+1)*T, 0);
    alphas[0]=1;
    for(int t=0;t<T;t++)
    {
        for(int u=0;u<=U;u++)
        {
            ut_null_index=(mb+minibatch*(t-1))*(maxU*3)+u;
            ut_forward_index=(mb+minibatch*(t-1))*(maxU*3)+u;
            if(t>0)
                alpha[t][u] += alpha[t-1][u]*probs_ut[ut_null_index];
            if(u>0)
                alpha[t][u] += alpha[t][u-1]*probs_ut[ut_forward_index];
        }
    }
    return alpha[T][U]
} 
template<typename ProbT>
ProbT CpuTransducer<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                            ProbT log_partition, int repeats,
                                            int S, int T, const int* const e_inc,
                                            const int* const s_inc,
                                            const int* const labels,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output) {
    for(int t=0;t<T;t++)
    {
    for(int u=0;u<U;u++)
    {
    
    }
    }

}
CpuCTC<ProbT>::cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                    const int* const labels,
                                    int T, int L, int mb, size_t bytes_used) {
    {
    //to do
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
     ProbT* trans_exp=static_cast<ProbT *>(workspace_);
     int maxT =*std::max_element(input_lengths,input_lengths+minibatch_);
     int maxU=*std::max_element(label_lengths,label_lengths+minibatch_);
     int trans_used= minibatch_ * alphabet_size_ * maxT;
     ProbT* predict_exp=trans_exp+trans_used;
     int predict_used=minibatch_ * alphabet_size_ * maxU;
     ProbT* probs_ut=predict_exp+predict_used;
     int probs_used=minibatch_*maxT*maxU*alphabet_size;
     size_t bytes_used+=(trans_used+predict_used+probs_used)*sizeof(ProbT);

     exp_matrix(trans_act,predict_act,trans_exp,predict_exp,input_lengths,label_lengths);
     compute_pr(trans_exp,predict_exp,input_lengths,label_lengths,maxT,maxU,flat_labels);
     for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int U = label_lengths[mb]; // Number of labels in transcription
        Cputransducer_metadata transducerm(L, S, T, mb, alphabet_size_, workspace_,
                             bytes_used + mb * per_minibatch_bytes, blank_label_,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));
         costs[mb] = -compute_alphas(probs_ut + mb * alphabet_size_,transducerm.alphas);
        }

    }

    return CTC_STATUS_SUCCESS;
}
