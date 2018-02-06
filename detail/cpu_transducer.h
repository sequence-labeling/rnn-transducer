#include <tuple>
#include <limits>
#include <numeric>
#include <transducer_helper.h>
template<typename ProbT>
void
CupTransducer<ProbT>::softmax(const ProbT* const act1,const ProbT* const act2,ProbT* probs,const int* const input_lengths,const int* const label_lengths)
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
