#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>
#include <transducer.h>
#include "test.h"
bool small_test()
{
const int alphabet_size=5;
const int T=2;
std::vector<float> trans_act={0.1, 0.9, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.9, 0.1, 0.1};

std::vector<float> predict_act={0.1, 0.9, 0.1, 0.1, 0.1,
                                       0.9, 0.2, 0.9, 0.1, 0.1,
                                         0.9, 0.1, 0, 0.1, 0.5};
std::vector<int> labels={1,2};
std::vector<int> label_lengths={2};
std::vector<int> lengths;
lengths.push_back(T);
float expected_score;
float score;
transducerOptions options{};
options.loc=TRANSDUCER_CPU;
options.num_threads=1;
size_t cpu_alloc_bytes;
 throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),alphabet_size, lengths.size(), options,&cpu_alloc_bytes),"Error: get_workspace_size in small_test");
    std::cout<<cpu_alloc_bytes<<std::endl;

 void* transducer_cpu_workspace = malloc(cpu_alloc_bytes);
throw_on_error(compute_transducer_loss(trans_act.data(),predict_act.data(), NULL,NULL,labels.data(), lengths.data(),
                                    label_lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score,
                                    transducer_cpu_workspace,
                                    options),
                   "Error: compute_transducer_loss in small_test");
free(transducer_cpu_workspace);
//score=std::exp(-score);
std::cout<<score<<std::endl;
const float eps=1e-6;
const float lb = expected_score - eps;
const float ub = expected_score + eps;
    return (score > lb && score < ub);

}
bool options_test()
 {
 const int alphabet_size=5;
 const int T=2;
 std::vector<float> trans_act={0.1, 0.9, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.9, 0.1, 0.1};

 std::vector<float> predict_act={0.1, 0.9, 0.1, 0.1, 0.1,
                                        0.9, 0.2, 0.9, 0.1, 0.1,
                                          0.9, 0.1, 0, 0.1, 0.5};
 std::vector<int> labels={1,2};
 std::vector<int> label_lengths={2};
 std::vector<int> lengths;
 lengths.push_back(T);
 float expected_score;
 float score;
 transducerOptions options{};
 options.loc=TRANSDUCER_CPU;
 options.num_threads=1;
 size_t cpu_alloc_bytes;
 std::vector<float> grads_trans(alphabet_size* 2);
 std::vector<float> grads_predict(alphabet_size* 3);

   throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),alphabet_size, lengths.size(), options,&cpu_alloc_bytes),"Error: get_workspace_size in small_test");
   std::cout<<cpu_alloc_bytes<<std::endl;
  void* transducer_cpu_workspace = malloc(cpu_alloc_bytes);
 throw_on_error(compute_transducer_loss(trans_act.data(),predict_act.data(),grads_trans.data(),grads_predict.data(),labels.data(), lengths.data(),
                                     label_lengths.data(),
                                     alphabet_size,
                                     lengths.size(),
                                     &score,
                                     transducer_cpu_workspace,
                                     options),
                    "Error: compute_transducer_loss in small_test");
 free(transducer_cpu_workspace);
 //score=std::exp(-score);
 std::cout<<score<<std::endl;
 for (auto i = grads_trans.begin(); i != grads_trans.end(); ++i)
    std::cout << *i << ' ';
 std::cout<<std::endl;
 for (auto i = grads_predict.begin(); i != grads_predict.end(); ++i)
    std::cout << *i << ' ';
 const float eps=1e-6;
 const float lb = expected_score - eps;
 const float ub = expected_score + eps;
     return (score > lb && score < ub);

 }


int main(void)
{    if (get_transducer_version() != 1) {
        std::cerr << "Invalid rnn transducer version." << std::endl;
        return 1;
    }
   std::cout << "Running CPU tests" << std::endl;
   bool status=true;
   status &= small_test();
   status &= options_test();
   if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}

