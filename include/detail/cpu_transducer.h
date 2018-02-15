#pragma once

#include <tuple>
#include <limits>
#include <numeric>
#include <iostream>
#include <algorithm>
#if !defined(TRANSDUCER_DISABLE_OMP)&&!defined(APPLE)
#include <omp.h>
#endif
#include "transducer_helper.h"
template <typename ProbT>
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
    transducerStatus_t cost_and_grad(const ProbT* const predict_acts, const ProbT* const trans_acts,
                              ProbT *predict_grads,ProbT *trans_grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const input_lengths,
                              const int* const label_lengths);
    transducerStatus_t score_forward(const ProbT* const trans_acts, const ProbT* const predict_acts,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const input_lengths,
                              const int* const label_lengths);
    private:
    class CpuTransducer_metadata
    {
        private:
         void  setup_probs(const ProbT* const trans_exp,const ProbT* const predict_exp,ProbT* probs_tuk,int T,int U,int alphabet_size,int minibatch,int mb);
        public:
            CpuTransducer_metadata(const ProbT* const trans_exp,const ProbT * const predict_exp,int T, int U, int alphabet_size,int minibatch,int mb,
                        void* workspace, size_t bytes_used);
     ProbT* alphas;
     ProbT* betas;
     ProbT* probs_tuk;
     ProbT* grads_tuk;
    };
    int alphabet_size_; // Number of characters plus null label
    int minibatch_;
    int num_threads_;
    int null_label_;
    void* workspace_;
    void exp_matrix(const ProbT* const trans_act,const ProbT*  const predict_probs,ProbT* trans_exp,ProbT* predict_exp,const int* const input_lengths,const int* const  label_lengths);
    ProbT compute_alphas(const ProbT* const probs_tuk, ProbT *const alphas,int T,int U,const int * label);
    ProbT compute_betas_and_grad(ProbT * trans_grads,ProbT * predict_grads, const ProbT* const probs_tuk,ProbT* const grads_tuk,const int * const label,int U, int T,ProbT* alphas,ProbT* betas);
    std::pair<ProbT,bool> cost_and_grad_kernel(ProbT *grads_trans,ProbT * grads_predict, const ProbT* const trans_exp,const ProbT* predict_exp,const int* const labels,int T, int U,int  alphabet_size,int minibatch,int mb, size_t bytes_used);
};
template<typename ProbT>
CpuTransducer<ProbT>::CpuTransducer_metadata::CpuTransducer_metadata(const ProbT* const trans_exp,const ProbT* const predict_exp,int T, int U, int alphabet_size,int minibatch,int mb,void* workspace, size_t bytes_used) {

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * U * T;
    std::fill(alphas, alphas + U* T,0);
    betas=reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
   bytes_used += sizeof(ProbT) * U ;
    std::fill(betas,betas+U,0);
    probs_tuk = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) *T*U*alphabet_size;
    grads_tuk=reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used+=sizeof(ProbT) *T*U*alphabet_size;
    setup_probs(trans_exp,predict_exp,probs_tuk,T,U,alphabet_size,minibatch,mb);
}
template<typename ProbT>
void CpuTransducer<ProbT>::CpuTransducer_metadata::setup_probs(const ProbT* const trans_exp,const ProbT* const predict_exp,ProbT* probs_tuk,int T,int U,int alphabet_size,int minibatch,int mb)
{
 ProbT sum;
 int tuk_index,trans_col_offset,predict_col_offset;
 for(int t=0;t<T;++t) 
 {
    trans_col_offset=(mb+minibatch*t)*alphabet_size;
    for(int u=0;u<U;++u)
        {
          tuk_index=(u+t*U)*alphabet_size;
          predict_col_offset=(mb+minibatch*u)*alphabet_size;
          sum=0;
          for(int r=0;r<alphabet_size;r++)
          {
            ProbT tmp=trans_exp[trans_col_offset+r]*predict_exp[predict_col_offset+r];
            probs_tuk[tuk_index+r]=tmp;
            sum+=tmp;
          }
          for(int r=0;r<alphabet_size;r++)
          {
            probs_tuk[tuk_index+r]/=sum;
          }
        }
 }
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
/*
template<typename ProbT>
CpuTransducer<ProbT>::cost_and_grad(const ProbT* const activations,
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
ProbT CpuTransducer<ProbT>::compute_alphas(const ProbT* const probs_tuk,ProbT * const alphas,int T,int U,const int* label)
{
    alphas[0]=1;
    int tuk_index=0,tuk_null_index,tuk_forward_index;
    int alphabet_index=0;
    for(int t=0;t<T;t++)
    {   
        alphabet_index=t*U;
        for(int u=0;u<U;u++)
        {
            if(t>0)
            {
                int tuk_index_tmp=(t-1)*U*alphabet_size_;
                tuk_null_index=tuk_index_tmp+u*alphabet_size_+null_label_;
                alphas[alphabet_index+u] += alphas[(t-1)*(U)+u]*probs_tuk[tuk_null_index];
            }
            if(u>0)
               { 
                int tuk_index_tmp=t*U*alphabet_size_;
                tuk_forward_index=tuk_index_tmp+(u-1)*alphabet_size_+label[u-1];
                alphas[alphabet_index+u] += alphas[alphabet_index+u-1]*probs_tuk[tuk_forward_index];
              }
        }
    }
    tuk_null_index=(T-1)*U*alphabet_size_+(U-1)*alphabet_size_+null_label_;
    ProbT loglike=alphas[(T-1)*U+(U-1)]*probs_tuk[tuk_null_index];
    return loglike;
} 
template<typename ProbT>
ProbT CpuTransducer<ProbT>::compute_betas_and_grad(ProbT* trans_grads,ProbT* predict_grads, const ProbT* const probs_tuk,ProbT* const grads_tuk,const int * const label,int T, int U,ProbT* alphas,ProbT* betas) {
    std::fill(trans_grads, trans_grads + T*alphabet_size_, 0);
    std::fill(predict_grads, predict_grads + U*alphabet_size_, 0);
    int tuk_null_index=(T-1)*U*alphabet_size_+(U-1)*alphabet_size_+null_label_;
    std::fill(betas,betas+U,probs_tuk[tuk_null_index]);
    ProbT pr_yx=alphas[(T-1)*U+(U-1)]*probs_tuk[tuk_null_index];
    ProbT beta_tu=probs_tuk[tuk_null_index],beta_tu_=probs_tuk[tuk_null_index],beta_t_u=probs_tuk[tuk_null_index];
    for(int t=T-1;t>=0;t--)
     {
         int alphas_index=t*U;
         for(int u=U-1;u>=0;u--)
         {
             beta_tu_=beta_tu;
             beta_t_u=betas[u];
             int tu_index=t*U*alphabet_size_+u*alphabet_size_;
             beta_tu=0;
             if(t<T-1)
             {
                 beta_tu+=beta_t_u*probs_tuk[tu_index+null_label_];
             }
              if(u<U-1)
             {
                 beta_tu+=beta_tu_*probs_tuk[tu_index+label[u]];
             }
             if(t==T-1&&u==U-1)
             {
                 beta_tu=probs_tuk[tuk_null_index];
             }
              betas[u]=beta_tu;
             int grads_tuk_index=(u+t*U)*alphabet_size_;
             for(int k=0;k<alphabet_size_;k++)
              {  
                  if(k==null_label_)
                  {
                      grads_tuk[grads_tuk_index+k]=-(alphas[alphas_index+u]/pr_yx)*beta_t_u;
                  }
                  else if(k==label[u])
                  { 
                      grads_tuk[grads_tuk_index+k]=-(alphas[alphas_index+u]/pr_yx)*beta_tu_;
                 
                  }
                  else
                  {
                      grads_tuk[grads_tuk_index+k]=0;
                  }
             }
       }
    }
    for(int t=0;t<T;t++)
    {
      int trans_grads_index=t*alphabet_size_;
      for(int k=0;k<alphabet_size_;k++)
      { 
        for(int u=0;u<U;u++)
        {
         int tuk_index=(u+t*U)*alphabet_size_;

         for(int k_tmp=0;k_tmp<alphabet_size_;k_tmp++)
         {
             trans_grads[trans_grads_index+k]+=grads_tuk[tuk_index+k_tmp]*probs_tuk[tuk_index+k_tmp]*(k==k_tmp?1:0-probs_tuk[tuk_index+k]);
         }

        }
       }
    }
    for(int u=0;u<U;u++)
    {
      int predict_grads_index=u*alphabet_size_;
      for(int k=0;k<alphabet_size_;k++)
      {
       for(int t=0;t<T;t++)
       {
         int tuk_index=(u+t*U)*alphabet_size_;
         for(int k_tmp=0;k_tmp<alphabet_size_;k_tmp++)
         {
          predict_grads[predict_grads_index+k]+=grads_tuk[tuk_index+k_tmp]*probs_tuk[tuk_index+k_tmp]*(1-probs_tuk[tuk_index+k]);
         }
       }
      }
    }
   return beta_tu;
}
template<typename ProbT>
std::pair<ProbT,bool> CpuTransducer<ProbT>::cost_and_grad_kernel(ProbT *grads_trans,ProbT * grads_predict, const ProbT* const trans_exp,const ProbT* predict_exp,const int* const labels,int T, int U,int alphabet_size,int minibatch,int mb, size_t bytes_used) {
    {
   CpuTransducer_metadata transducerm(trans_exp,predict_exp,T,U, alphabet_size,minibatch,mb,workspace_, bytes_used);
    bool over_threshold = false;

    if (U-1> T) {
        return std::make_pair(ProbT(0), over_threshold); // TODO, not right to return 0
    }
    ProbT llForward = compute_alphas(transducerm.probs_tuk,transducerm.alphas,T,U,labels);
    ProbT llBackward = compute_betas_and_grad(grads_trans,grads_predict, transducerm.probs_tuk,transducerm.grads_tuk,labels,T,U,transducerm.alphas,transducerm.betas);
    ProbT diff = std::abs(llForward - llBackward);
    if (diff > transducer_helper::threshold) {
        over_threshold = true;
    }

    return std::make_pair(llBackward,over_threshold);
    }
}

template<typename ProbT>
transducerStatus_t CpuTransducer<ProbT>::cost_and_grad(const ProbT* const trans_acts, const ProbT* const predict_acts, ProbT* grads_trans,ProbT* grads_predict,ProbT* costs,const int* const flat_labels,const int* const input_lengths, const int* const label_lengths) 
 {
    if (trans_acts == nullptr ||
        predict_acts==nullptr||
        grads_trans == nullptr ||
        grads_predict==nullptr||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return TRANSDUCER_STATUS_INVALID_VALUE;

      ProbT* trans_exp=static_cast<ProbT *>(workspace_);
      int maxT =*std::max_element(input_lengths,input_lengths+minibatch_);
      int maxU=*std::max_element(label_lengths,label_lengths+minibatch_)+1;
      int trans_used= minibatch_ * alphabet_size_ * maxT;
      ProbT* predict_exp=trans_exp+trans_used;
      int predict_used=minibatch_ * alphabet_size_ * maxU;
      size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * (maxT+maxU);
      size_t per_minibatch_bytes=0;
      per_minibatch_bytes += sizeof(ProbT) * maxU * (maxT+1);
      per_minibatch_bytes += sizeof(ProbT) *  maxU * maxT*alphabet_size_;
      per_minibatch_bytes += sizeof(ProbT) *  maxU * maxT*alphabet_size_;
      exp_matrix(trans_acts,predict_acts,trans_exp,predict_exp,input_lengths,label_lengths);

   #pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int U = label_lengths[mb]+1; // Number of labels in transcription
        bool mb_status;

        std::tie(costs[mb], mb_status) =cost_and_grad_kernel(grads_trans,grads_predict,trans_exp,predict_exp,
                                     flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                                     T, U, alphabet_size_,minibatch_, mb,
                                     bytes_used + mb * per_minibatch_bytes);
    }

    return TRANSDUCER_STATUS_SUCCESS;
}
template<typename ProbT>
transducerStatus_t CpuTransducer<ProbT>::score_forward(const ProbT* const trans_acts, const ProbT* const predict_acts,
                                         ProbT* costs,
                                         const int* const  flat_labels,
                                         const int* const input_lengths,
                                         const int* const label_lengths) {
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
     ProbT* predict_exp=trans_exp+trans_used;
     int predict_used=minibatch_ * alphabet_size_ * maxU;
     size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * (maxT+maxU);
     size_t per_minibatch_bytes=0;
     per_minibatch_bytes += sizeof(ProbT) * maxU * (maxT+1);
     per_minibatch_bytes += sizeof(ProbT) *  maxU * maxT*alphabet_size_;
     per_minibatch_bytes += sizeof(ProbT) *  maxU * maxT*alphabet_size_;
     exp_matrix(trans_acts,predict_acts,trans_exp,predict_exp,input_lengths,label_lengths);
     //compute_pr(trans_exp,predict_exp,probs_utk,maxU,maxT,input_lengths,label_lengths);
     int T=0,U=0;
     for (int mb = 0; mb < minibatch_; ++mb) {
          const int T = input_lengths[mb]; // Length of utterance (time)
          const int U = label_lengths[mb]+1;
          CpuTransducer_metadata transducerm(trans_exp,predict_exp,T,U, alphabet_size_,minibatch_,mb,workspace_, bytes_used);
          costs[mb] = -compute_alphas(transducerm.probs_tuk,transducerm.alphas,T,U,flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));
        }


    return TRANSDUCER_STATUS_SUCCESS;
}
