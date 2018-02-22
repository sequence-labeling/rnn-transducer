//some c struct for transducer for CPU
#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif
typedef enum {
    TRANSDUCER_STATUS_SUCCESS = 0,
    TRANSDUCER_STATUS_MEMOPS_FAILED = 1,
    TRANSDUCER_STATUS_INVALID_VALUE = 2,
    TRANSDUCER_STATUS_EXECUTION_FAILED = 3,
    TRANSDUCER_STATUS_UNKNOWN_ERROR = 4
} transducerStatus_t;
typedef struct CUstream_st* CUstream;
typedef enum
{
TRANSDUCER_CPU=0,
TRANSDUCER_GPU=1
} transducerComputeLocation;
 struct transducerOptions
 {
 transducerComputeLocation loc;
 union
 {
 unsigned int num_threads;
 CUstream stream;
 };
 int null_label;
 
 };
const char* transducerGetStatusString(transducerStatus_t status);
int get_transducer_version();
transducerStatus_t compute_transducer_loss(const float * const trans_act,const float * const predict_act,float * trans_grad,float * predict_grad,const int * const flat_labels,const int * const input_lengths,const int * const label_lengths,int alphabet_size,int minibatch, float * costs, void * workspace ,transducerOptions options);
transducerStatus_t get_workspace_size(const int* const input_lengths,
                               const int* const label_lengths,
                               int alphabet_size, int minibatch,
                               transducerOptions info,
                               size_t* size_bytes);
#ifdef __cplusplus
}
#endif
