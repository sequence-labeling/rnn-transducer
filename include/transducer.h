//some c struct for transducer for CPU
typedef enum {
    TRANSDUCER_STATUS_SUCCESS = 0,
    TRANSDUCER_STATUS_MEMOPS_FAILED = 1,
    TRANSDUCER_STATUS_INVALID_VALUE = 2,
    TRANSDUCER_STATUS_EXECUTION_FAILED = 3,
    TRANSDUCER_STATUS_UNKNOWN_ERROR = 4
} transducerStatus_t;
transducerStatus_t compute_transducer_loss(const float * const predict_pro,float * gradients,const int * const flat_labels,const int * const label_lengths,const int * const input_lengths,int alphabet_size,int minibatch, flaot * costs, transducerOptions options)
