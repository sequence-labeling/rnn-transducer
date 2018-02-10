#pragma once
#include<stdexcept>
#include<vector>
#include<random>

#include<transducer.h>
inline void throw_on_error(transducerStatus_t status,const char * message)
{
if(status!=TRANSDUCER_STATUS_SUCCESS)
{

 throw std::runtime_error(message + (", stat = " + 
                                            std::string(transducerGetStatusString(status))));
}
}
std::vector<float>
genActs(int size) {
    std::vector<float> arr(size);
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(0, 1);
    for(int i = 0; i < size; ++i)
        arr[i] = dis(gen);
    return arr;
}

std::vector<int>
genLabels(int alphabet_size,int L)
{
    std::vector<int> label(L);
    std::mt19937 gen(1);
    std::uniform_int_distribution<>dis(1,alphabet_size+1);
    for(int i=0;i<L;++i)
    {
        label[i]=dis(gen);
    }
    return label;
}
float rel_diff(const std::vector<float>& grad,
               const std::vector<float>& num_grad) {
    float diff = 0.;
    float tot = 0.;
    for(size_t idx = 0; idx < grad.size(); ++idx) {
        diff += (grad[idx] - num_grad[idx]) * (grad[idx] - num_grad[idx]);
        tot += grad[idx] * grad[idx];
    }

    return diff / tot;
}
