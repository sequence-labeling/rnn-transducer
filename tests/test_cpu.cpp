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
void* transducer_cpu_workspace = malloc(cpu_alloc_bytes);
throw_on_error(compute_transducer_loss(trans_act.data(),predict_act.data(), NULL,NULL,labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score,
                                    transducer_cpu_workspace,
                                    options),
                   "Error: compute_transducer_loss in small_test");
free(transducer_cpu_workspace);
//score=std::exp(-score);
std::cout<<score;
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
   if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}

