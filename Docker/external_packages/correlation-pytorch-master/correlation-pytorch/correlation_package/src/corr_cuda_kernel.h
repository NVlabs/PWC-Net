#ifndef _CORR_CUDA_KERNEL
#define _CORR_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void blob_rearrange_ongpu(const float *in, float *out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight, cudaStream_t stream);

void CorrelateData_ongpu(const float *rbot1, const float *rbot2, float *output, int batchSize, int nOutputCols, int nOutputRows, int nOutputPlane, int max_displacement, int neighborhood_grid_radius_, int neighborhood_grid_width_, int kernel_radius_, int kernel_size, int stride1, int stride2, int paddedbottomwidth, int paddedbottomheight, int nInputPlane, int corr_type_multiply, cudaStream_t stream);

void CorrelateDataBackward_ongpu(const float *rbot1, const float *rbot2, const float *gradOutput, float *gradInput1, float *gradInput2, int batchSize, int nOutputCols, int nOutputRows, int nOutputPlane, int max_displacement, int neighborhood_grid_radius_, int neighborhood_grid_width_, int kernel_radius_, int stride1, int stride2, int nInputCols, int nInputRows, int paddedbottomwidth, int paddedbottomheight, int nInputPlane, int pad_size, int corr_type_multiply, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
