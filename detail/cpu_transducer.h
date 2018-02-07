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
CpuTransducer<ProbT>::CpuTransducer_metadata::CpuTransducer_metadata(int L, int S, int T, int mb,
                                                int alphabet_size,
                                                void* workspace, size_t bytes_used,
                                                int blank_label,
                                                const int* const labels) {

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S * T;
    std::fill(alphas, alphas + S * T, ctc_helper::neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S;
    std::fill(betas, betas + S, ctc_helper::neg_inf<ProbT>());
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
ProbT CpuTransducer<Prob>::compute_alphas(const ProbT* const predict_act,const ProbT* trans_act,int U,int * labels,int T,ProbT* alphas)
{
    std::fill(alphas, alphas + (U+1)*T, 0);
    alphas[0]=1;
    for(int t=0;t<T;t++)
    {
        for(int u=0;u<=U;u++)
        {
            if(t>0)
                alpha[t][u] += alpha[t-1][u]*O(t-1,u);
            if(u>0)
                alpha[t][u] += alpha[t][u-1]*y(t,u-1);
        }
    }
    return alpha[T][U]
} 
template<typename Prob>
void
CpuTransduer<Prob>::compute_pr(const ProbT* const predict_act,const ProbT* trans_act,int k,int t,int u,int U,int *labels)
{
    ProbT sum=0,prob_null,prob_back,prob_forword;
    probs_utk.clear();
    for(int c=0;c<alphabet_size_;c++)
    {   
        ProbT tmp=trans_act[t*alphabet_size_+c]*predict_act[u*alphabet_size_+c];
        if(c==null_label_)
            prob_null=tmp;
        else if(u+1<=U&&c==label[u+1])
            prob_back=tmp;
        else if(u-1>=0&&c==label[u-1]
            prob_forword=tmp;
        sum+=tmp;
    }
    if(u+1<=U)
        probs_utk[make_pair(u,make_pair(t,label[u+1]))]=prob_back/sum;
    if(u-1>=0)
        probs_utk[macke_pair(u,make_pair(t,label[u-1]))]=prob_forword/sum;
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
        exp_matrix(predict_probs,trans_probs,input_lengths,label_lengths);
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int U = label_lengths[mb]; // Number of labels in transcription

       compute_pr(predict_act,trans_act,T,U,int *labels)

         costs[mb] = -compute_alphas(probs + mb * alphabet_size_, ctcm.repeats, S, T,
                                        ctcm.e_inc, ctcm.s_inc, ctcm.labels_w_blanks,
                                        ctcm.alphas);
        }

    }

    return CTC_STATUS_SUCCESS;
}
