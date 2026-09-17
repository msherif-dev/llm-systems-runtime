#pragma once

void reduce_sum_cuda(
    const float* d_input,
    float* d_output,
    int N
);
