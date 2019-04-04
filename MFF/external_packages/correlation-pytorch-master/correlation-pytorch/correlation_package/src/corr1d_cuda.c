#include <THC/THC.h>
#include "corr1d_cuda_kernel.h"

extern THCState *state;

// == Forward
int corr1d_cuda_forward(THCudaTensor *input1,
                      THCudaTensor *input2,
                      THCudaTensor *rbot1,
                      THCudaTensor *rbot2,
                      THCudaTensor *output,
                      int pad_size,
                      int kernel_size,
                      int max_displacement,
                      int stride1,
                      int stride2,
                      int corr_type_multiply
                      //single_direction=0 
                      )
{

    // TODO: Shapechecks

    int batchSize = input1->size[0];

    long nInputPlane = input1->size[1];
    long nInputRows = input1->size[2];
    long nInputCols = input1->size[3];
    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    int x_shift = -neighborhood_grid_radius_;

    // Number of output channels amounts to displacement combinations in X direction only!!
    int nOutputPlane = neighborhood_grid_width_;//Same, because 1D X-correlation 

    // Inputs
    float * input1_data = THCudaTensor_data(state, input1);
    float * input2_data = THCudaTensor_data(state, input2);

    // Outputs
    THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, nOutputRows, nOutputCols);
    THCudaTensor_zero(state, output); // added by Jinwei
    float * output_data = THCudaTensor_data(state, output);

    THCudaTensor_resize4d(state, rbot1, batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth);
    THCudaTensor_resize4d(state, rbot2, batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth);

    THCudaTensor_zero(state, rbot1); // added by Jinwei
    THCudaTensor_zero(state, rbot2); // added by Jinwei

    float * rbot1_data = THCudaTensor_data(state, rbot1);
    float * rbot2_data = THCudaTensor_data(state, rbot2);

    cudaStream_t stream = THCState_getCurrentStream(state);

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    blob_rearrange_ongpu_1d(input1_data,rbot1_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    blob_rearrange_ongpu_1d(input2_data,rbot2_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    CorrelateData_ongpu_1d(rbot1_data,rbot2_data,output_data,batchSize,nOutputCols,nOutputRows,nOutputPlane,max_displacement,x_shift,neighborhood_grid_width_,kernel_radius_,kernel_size,stride1,stride2,paddedbottomwidth,paddedbottomheight,nInputPlane,corr_type_multiply,stream);

//    THCudaTensor_free(state, input1);
//    THCudaTensor_free(state, input2);
    THCudaTensor_free(state, rbot1);
    THCudaTensor_free(state, rbot2);

    return 1;

}

int corr1d_cuda_backward(THCudaTensor *input1,
                        THCudaTensor *input2,
                        THCudaTensor *rbot1,
                        THCudaTensor *rbot2,
                        THCudaTensor *gradOutput,
                        THCudaTensor *gradInput1,
                        THCudaTensor *gradInput2,
                        int pad_size,
                        int kernel_size,
                        int max_displacement,
                        int stride1,
                        int stride2,
                        int corr_type_multiply
                        // single_direction=0
                        )
{

    float * input1_data = THCudaTensor_data(state, input1);
    float * input2_data = THCudaTensor_data(state, input2);

    long nInputCols = input1->size[3];
    long nInputRows = input1->size[2];
    long nInputPlane = input1->size[1];
    long batchSize = input1->size[0];

 //   THCudaTensor_resizeAs(state, gradInput1, input1);
 //   THCudaTensor_resizeAs(state, gradInput2, input2);
    float * gradOutput_data = THCudaTensor_data(state, gradOutput);
    float * gradInput1_data = THCudaTensor_data(state, gradInput1);
    float * gradInput2_data = THCudaTensor_data(state, gradInput2);

    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    int x_shift = -neighborhood_grid_radius_;

    // Number of output channels amounts to displacement combinations in X direction only!!
    int nOutputPlane = neighborhood_grid_width_; // Same, because 1D X-correlation

    THCudaTensor_resize4d(state, rbot1, batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth);
    THCudaTensor_resize4d(state, rbot2, batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth);

    THCudaTensor_zero(state, rbot1); // added by Jinwei
    THCudaTensor_zero(state, rbot2); // added by Jinwei

    float * rbot1_data = THCudaTensor_data(state, rbot1);
    float * rbot2_data = THCudaTensor_data(state, rbot2);

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    cudaStream_t stream = THCState_getCurrentStream(state);

    blob_rearrange_ongpu_1d(input1_data,rbot1_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    blob_rearrange_ongpu_1d(input2_data,rbot2_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    // CorrelationLayerBackward

    CorrelateDataBackward_ongpu_1d(rbot1_data,rbot2_data,gradOutput_data,gradInput1_data,gradInput2_data,batchSize,nOutputCols,nOutputRows,nOutputPlane,max_displacement,x_shift,neighborhood_grid_width_,kernel_radius_,stride1,stride2,nInputCols,nInputRows,paddedbottomwidth,paddedbottomheight,nInputPlane,pad_size,corr_type_multiply,stream);

  //  THCudaTensor_free(state, input1);
  //  THCudaTensor_free(state, input2);
    THCudaTensor_free(state, rbot1);
    THCudaTensor_free(state, rbot2);

    return 1;

}
