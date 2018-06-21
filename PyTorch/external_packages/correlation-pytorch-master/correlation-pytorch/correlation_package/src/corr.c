#include <TH/TH.h>

int corr_cpu_forward(THFloatTensor *input1,
                      THFloatTensor *input2,
                      THFloatTensor *rbot1,
                      THFloatTensor *rbot2,
                      THFloatTensor *output,
                      int pad_size,
                      int kernel_size,
                      int max_displacement,
                      int stride1,
                      int stride2,
                      int corr_type_multiply)
{
    return 1;
}

int corr_cpu_backward(THFloatTensor *input1,
                       THFloatTensor *input2,
                       THFloatTensor *rbot1,
                       THFloatTensor *rbot2,
                       THFloatTensor *gradOutput,
                       THFloatTensor *gradInput1,
                       THFloatTensor *gradInput2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{
    return 1;
}

