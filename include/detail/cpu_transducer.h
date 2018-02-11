#include <tuple>
#include <limits>
#include <numeric>
#include <iostream>
#if !defined(TRANSDUCER_DISABLE_OMP)&&!defined(APPLE)
#include<omp.h>
#endif
#include "transducer_helper.h"
template<typename ProbT>
class CpuTransducer
{
public:
    CpuTransducer(int alphabet_size, int minibatch, void* workspace, int num_threads,
           int null_label) :
            alphabet_size_(alphabet_size), minibatch_(minibatch),
            num_threads_(num_threads), workspace_(workspace),
            null_label_(null_label) {
 #if defined TRANSDUCER_DISABLE_OMP || defined(APPLE)
 #else
         if(num_threads>0)
         {
         omp_set_threads(num_threads);
         }else
         {
         num_threads_=omp_get_max_threads();
         }
 #endif
     };
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
        public:
            CpuTransducer_metadata(int U, int T, int mb, int alphabet_size,
                        void* workspace, size_t bytes_used, int null_label,
                        const int* const labels);
     ProbT* alphas;
     ProbT* betas;
     ProbT* output;
    };
    int alphabet_size_; // Number of characters plus null label
    int minibatch_;
    int num_threads_;
    int null_label_;
    void* workspace_;
    void exp_matrix(const ProbT* const trans_act,const ProbT*  const predict_probs,ProbT* trans_exp,ProbT* predict_exp,const int* const input_lengths,const int* const  label_lengths);
    void compute_pr(const ProbT* const trans_exp,const ProbT* predict_exp,ProbT* const probs_utk,int maxU,int maxT,const int * const input_lengths,const int * const    label_lengths);
      ProbT compute_alphas(const ProbT* const probs_tuk, ProbT *const alphas, int maxT,int max_U,int T,int U,const int * label);

        ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                 ProbT log_partition, int repeats,
                                 int S, int T, const int* const e_inc,
                                 const int* const s_inc,
                                 const int* const labels,
                                 ProbT* alphas,
                                 ProbT* betas,
                                 ProbT* output);

};
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
    //std::fill(betas, betas + S, 1);
    //labels_w_blanks = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    //bytes_used += sizeof(int) * S;
    //e_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
   // bytes_used += sizeof(int) * S;
   // s_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    //bytes_used += sizeof(int) * S;
    output = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * alphabet_size;

}
template<typename ProbT>
void
CpuTransducer<ProbT>::exp_matrix(const ProbT* const trans_acts,const ProbT*  const predict_acts,ProbT* trans_exp,ProbT* predict_exp,const int* const input_lengths,const int* const label_lengths)
{
 for (int mb = 0; mb < minibatch_; ++mb) {
      for(int c=0;c<input_lengths[mb];++c)
        {
            for(int r = 0; r < alphabet_size_; ++r)
             {
                int col_offset = (mb + minibatch_ * c) * alphabet_size_;
                trans_exp[r + col_offset] = std::exp(trans_acts[r+col_offset]);
              }

          }
      for(int c=0;c<label_lengths[mb]+1;c++)
      {
           for(int r = 0; r < alphabet_size_; ++r)
               {
                 int col_offset = (mb + minibatch_ * c) * alphabet_size_;
                 predict_exp[r + col_offset] = std::exp(predict_acts[r+ col_offset]);
               }
      }
}
}
template<typename ProbT>
void
CpuTransducer<ProbT>::compute_pr(const ProbT* const trans_exp,const ProbT* predict_exp,ProbT* const probs_utk,int maxU,int maxT,const int * const input_lengths,const int * const label_lengths)
{
    // maxT*maxU*mini_batch_*alphabet_size_;
   ProbT sum=0;
   int utk_index,trans_col_offset,predict_col_offset; 
   for(int mb=0;mb<minibatch_;++mb)
   {
     for(int t=0;t<input_lengths[mb];++t)
     {
        trans_col_offset=(mb+minibatch_*t)*alphabet_size_;
       for(int u=0;u<label_lengths[mb]+1;++u)
       {  
           utk_index=(mb+u*minibatch_+t*maxU*minibatch_)*alphabet_size_;
         predict_col_offset=(mb+minibatch_*u)*alphabet_size_;
         sum=0;
         for(int r=0;r<alphabet_size_;r++)
         {
           ProbT tmp=trans_exp[trans_col_offset+r]*predict_exp[predict_col_offset+r];
           probs_utk[utk_index+r]=tmp;
           sum+=tmp;
         }
         for(int r=0;r<alphabet_size_;r++)
         {
           probs_utk[utk_index+r]/=sum;
         }
       }
     }
   }
}
/*template<typename ProbT>
transducer<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths
)
{
//if (activations == nullptr ||
  //      grads == nullptr ||
    //    costs == nullptr ||
      //  flat_labels == nullptr ||
        //label_lengths == nullptr ||
        //input_lengths == nullptr
       // )
        return TRANSDUCER_STATUS_INVALID_VALUE;
//todo
}
*/
//t,u   u*T+t
//u: [0,U]
//t: [0,T)
//must invoke compute_pr before invoke this function
template<typename ProbT>
ProbT CpuTransducer<ProbT>::compute_alphas(const ProbT* const probs_tuk,ProbT * const alphas,int maxT,int maxU,int T,int U,const int* label)
{
    std::fill(alphas, alphas + U*T, 0);
    alphas[0]=1;
    int tuk_index=0,tuk_null_index,tuk_forward_index;
    int alphabet_index=0;
    //std::cout<<"null_label"<<null_label_;
    for(int t=0;t<T;t++)
    {   
        alphabet_index=t*U;
        for(int u=0;u<U;u++)
        {
            if(t>0)
            {
                int tuk_index_tmp=(t-1)*maxU*minibatch_*alphabet_size_;
                tuk_null_index=tuk_index_tmp+u*minibatch_*alphabet_size_+null_label_;
                alphas[alphabet_index+u] += alphas[(t-1)*(U)+u]*probs_tuk[tuk_null_index];
            }
            if(u>0)
               { 
                int tuk_index_tmp=t*maxU*minibatch_*alphabet_size_;
                tuk_forward_index=tuk_index_tmp+(u-1)*minibatch_*alphabet_size_+label[u-1];
                alphas[alphabet_index+u] += alphas[alphabet_index+u-1]*probs_tuk[tuk_forward_index];
              }
        }
       // std::cout<<"hello word";
    }
    tuk_null_index=(T-1)*maxU*minibatch_*alphabet_size_+(U-1)*minibatch_*alphabet_size_+null_label_;
    ProbT loglike=alphas[(T-1)*U+(U-1)]*probs_tuk[tuk_null_index];
    return loglike;
} 
/*
template<typename ProbT>
ProbT CpuTransducer<ProbT>::compute_betas_and_grad(ProbT* trans_grad,ProbT* predict_grad, const ProbT* const probs_ut,
                                            ProbT log_partition, int repeats,
                                            int U, int T, const int* const e_inc,
                                            const int* const s_inc,
                                            const int* const labels,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output) {
    pr_yx=
    for(int k=0;k<alphabet_size_;k++)
    {
    
    for(int t=0;t<T;t++)
     {
         for(int u=0;u<=U;u++)
         {
          grad_utk=-alphas[u][t]/pr_yx
         }
     }
    }
    for(int t=0,t<T;t++)
    {
    for(int k=0,k<alphabet_size_,k++)
    {
    for(int u=0;u<U;u++)
    {
         for(int k_tmp=0;k_tmp<alphabet_size;k_tmp++)
         {
             trans_grad[k][t]+=grad_utk[u][t][k_tmp]*porbs_utk[u][t][k_tmp]*(//totp);
         }

    }
    trans_grad[k][t]=
    }

    }
    for(int u=0;u<U;u++)
    {
    for(int k=0;k<alphabeet_size_,k++)
    {
    for(int t=0;t<T;t++)
    {
      for(int k_tmp=0;k_tmp<alphabet_size;k_tmp++)
      {
         predit_grad[k][u]+=grad_utk[u][t][k_tmp]*porbs_utk[u][t][k_tmp]*(//totp);
      }
    }
    }
    }
   return pr_yx;
}
CpuCTC<ProbT>::cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                    const int* const labels,
                                    int T, int L, int mb, size_t bytes_used) {
    {
    //to do
    }
*/
template<typename ProbT>
transducerStatus_t CpuTransducer<ProbT>::score_forward(const ProbT* const trans_acts, const ProbT* const predict_acts,
                                         ProbT* costs,
                                         const int const * flat_labels,
                                         const int* const label_lengths,
                                         const int* const input_lengths) {
    if (predict_acts == nullptr ||
        trans_acts==nullptr||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return TRANSDUCER_STATUS_INVALID_VALUE;
     //node:! trans_exp is  maxT*mini_batch_*alalphabet_size
     ProbT* trans_exp=static_cast<ProbT *>(workspace_);
     int maxT =*std::max_element(input_lengths,input_lengths+minibatch_);
     int maxU=*std::max_element(label_lengths,label_lengths+minibatch_)+1;
     int trans_used= minibatch_ * alphabet_size_ * maxT;
     //preddict is maxU*mini_bach_*alphabet
     ProbT* predict_exp=trans_exp+trans_used;
     int predict_used=minibatch_ * alphabet_size_ * maxU;
     ProbT* probs_utk=predict_exp+predict_used;
     int probs_used=minibatch_*maxT*maxU*alphabet_size_;
     //probs is maxT*maxU*mini_batch_*alphabet_size_;
     ProbT* alphas=probs_utk+probs_used;
     exp_matrix(trans_acts,predict_acts,trans_exp,predict_exp,input_lengths,label_lengths);
     compute_pr(trans_exp,predict_exp,probs_utk,maxU,maxT,input_lengths,label_lengths);
     const int *label=flat_labels;
     int T=0,U=0;
     for (int mb = 0; mb < minibatch_; ++mb) {
          alphas+=T*U;
          label+=U;
         const int T = input_lengths[mb]; // Length of utterance (time)
         const int U = label_lengths[mb]+1; // Number of labels in transcription
         costs[mb] = -compute_alphas(probs_utk + mb * alphabet_size_,alphas,maxT,maxU,T,U,label);
        }


    return TRANSDUCER_STATUS_SUCCESS;
}
