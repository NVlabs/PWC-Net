#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "corr1d_cuda_kernel.h"

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t

// == Dimension rearrangement Kernel
__global__ void blob_rearrange_kernel2_1d(const float *in, float *out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;


    float value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + 0);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}

void blob_rearrange_ongpu_1d(const float *in, float *out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight, cudaStream_t stream)
{
    int threads_per_block=16;
    dim3 totalBlocksRearr((widthheight-1)/threads_per_block+1, channels, num);

    cudaError_t err;

    blob_rearrange_kernel2_1d<<<totalBlocksRearr, threads_per_block, 0, stream>>>
        (in, out, num, channels, width, height, widthheight, padding, pwidthheight);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// == Correlation Kernel

__global__ void CorrelateData_1d(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const float *bottom0, const float *bottom1, float *top)
{
  extern __shared__ char patch_data_char[];
  
  float *patch_data = (float *)patch_data_char;
  
    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  
  // Load 3D patch into shared shared memory
  for(int j = 0; j < kernel_size; j++) { // HEIGHT
    for(int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }
  
  __syncthreads();
  
  __shared__ float sum[WARPS_PER_BLOCK*THREADS_PER_WARP];

  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;
  
    int s2o = (top_channel % neighborhood_grid_width + x_shift) * stride2;
    
    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int x2 = x1 + s2o;
          
          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y1+j) * bottomwidth + x2+i) * bottomchannels + ch;
          //int idx2 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          
          //printf("x1 %d x2 %d bh %d bw %d bc %d i %d ch %d y1 %d idx2 %d\n", x1, x2, bottomheight, bottomwidth, bottomchannels, item, ch, y1, idx2);

          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }

    __syncthreads();
    
    if(ch_off == 0) {
        float total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        //printf("ch_off %d sum %f\n", ch_off, total_sum);
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }
  
  
  // Aggregate  
}

__global__ void CorrelateDataSubtract_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const float *bottom0, const float *bottom1, float *top) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % topwidth; //w-pos
    int y = (index / topwidth) % topheight; //h-pos
    int c = (index / topwidth / topheight) % topchannels; //channels
        
    // Offset of patch in image 2
    int s2o = (c % neighborhood_grid_width + x_shift) * stride2;
        
    // First (upper left) position of kernel center in current neighborhood in image 1
    int x1 = x*stride1 + kernel_radius + max_displacement;
    int y1 = y*stride1 + kernel_radius + 0;
    
    // Iterate through 3D patch
    float sum = 0;
    for(int j = -kernel_radius; j <= kernel_radius; j++) { // HEIGHT
      for(int i = -kernel_radius; i <= kernel_radius; i++) { // WIDTH
        for(int l = 0; l < bottomchannels; l++) { // CHANNELS
          // Calculate position in image 2
          int x2 = x1 + s2o;
          int y2 = y1;

          // Indices in bottom data: (CH=l,W=x2,H=y2,N)
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + l;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + l;

          // Do the correlation:
          sum += fabsf(bottom0[idx1] - bottom1[idx2]);
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    top[index + item*topcount] = sum / (float)sumelems;
  }

}

void CorrelateData_ongpu_1d(const float *rbot1, const float *rbot2, float *output, int batchSize, int nOutputCols, int nOutputRows, int nOutputPlane, int max_displacement, int x_shift, int neighborhood_grid_width_, int kernel_radius_, int kernel_size, int stride1, int stride2, int paddedbottomwidth, int paddedbottomheight, int nInputPlane, int corr_type_multiply, cudaStream_t stream)
{

    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);

    int shared_memory_per_block = (kernel_size*kernel_size)*nInputPlane;

    int outputCount = nOutputCols * nOutputRows * nOutputPlane;
    int outputThreadCount = outputCount;

    if (corr_type_multiply == 1) {

        dim3 totalBlocksCorr(nOutputCols, nOutputRows, batchSize);

        CorrelateData_1d<<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(float), stream>>>(
                outputThreadCount,
                batchSize, nOutputCols, nOutputRows, nOutputPlane, outputCount,
                max_displacement, x_shift,
                neighborhood_grid_width_, kernel_radius_, kernel_size,
                stride1, stride2,
                paddedbottomwidth, paddedbottomheight, nInputPlane,
                rbot1, rbot2, output
                );
    } else {

        for (int n = 0; n < batchSize; n++) {

            CorrelateDataSubtract_1d<<<GET_BLOCKS(outputThreadCount, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
                    outputThreadCount,
                    batchSize, n, nOutputCols, nOutputRows, nOutputPlane, outputCount,
                    max_displacement, x_shift, neighborhood_grid_width_,
                    kernel_radius_, stride1, stride2,
                    paddedbottomwidth, paddedbottomheight, nInputPlane,
                    rbot1, rbot2, output
                    );
        }
    }
}

// == Correlation Backward Pass Kernel (For Blob 0)

__global__ void CorrelateDataBackward0_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  float *bottom0diff, const float *bottom1, const float *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    float sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        {
          for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int idxbot1 = ((item * pbottomheight + m) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            float bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m,n]

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot0index = ((n * bottomheight) + m) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}

// == Correlation Backward Pass Kernel (For Blob 1)
__global__ void CorrelateDataBackward1_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const float *bottom0, float *bottom1diff, const float *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    //int l = index % bottomwidth + pad_size; //w-pos
    //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight; //h-pos
    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    float sum = 0;
    {
      for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {
        
        int s2o = stride2 * o;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + m) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m,n]

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
		const int bot1index = ((n * bottomheight) + m) * bottomwidth + (l-pad_size);
		bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}

// == Correlation Kernel Subtraction


// == Correlation Backward Pass Kernel (For Blob 0)
__global__ void CorrelateDataBackward0Subtract_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  float *bottom0diff, const float *bottom0, const float *bottom1, const float *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    float sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        {
          for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
            float bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
            float sign = (bot0tmp >= bot1tmp) ? float(1.0) : float(-1.0);

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom0diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}


// == Correlation Backward Pass Kernel (For Blob 1)
__global__ void CorrelateDataBackward1Subtract_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const float *bottom0, const float *bottom1, float *bottom1diff, const float *topdiff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int l = index % bottomwidth + pad_size; //w-pos
    int m = (index / bottomwidth) % bottomheight; //h-pos
    int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    float sum = 0;
    {
      for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {
        
        int s2o = stride2 * o;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
            float bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
            float sign = (bot0tmp >= bot1tmp) ? float(-1.0) : float(1.0);

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    bottom1diff[index + item*bottomcount] = sum / (float)sumelems;
  }

}

void CorrelateDataBackward_ongpu_1d(const float *rbot1, const float *rbot2, const float *gradOutput, float *gradInput1, float *gradInput2, int batchSize, int nOutputCols, int nOutputRows, int nOutputPlane, int max_displacement, int x_shift, int neighborhood_grid_width_, int kernel_radius_, int stride1, int stride2, int nInputCols, int nInputRows, int paddedbottomwidth, int paddedbottomheight, int nInputPlane, int pad_size, int corr_type_multiply, cudaStream_t stream)
{
    int inputCount = nInputPlane * nInputRows * nInputCols;
    int botThreadCount = inputCount;

    if (corr_type_multiply == 1) {

        // == Run kernel Backward 0
        for (int n = 0; n < batchSize; n++) {
            //Bottom0
            CorrelateDataBackward0_1d<<<GET_BLOCKS(botThreadCount, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
                    botThreadCount,
                    batchSize, n, nOutputCols, nOutputRows, nOutputPlane,
                    max_displacement, x_shift, neighborhood_grid_width_,
                    kernel_radius_, stride1, stride2, nInputCols, nInputRows,
                    paddedbottomwidth, paddedbottomheight, nInputPlane, inputCount, pad_size,
                    gradInput1, rbot2, gradOutput
                    );

        }

        // == Run kernel Backward 1
        for (int n = 0; n < batchSize; n++) {
            CorrelateDataBackward1_1d<<<GET_BLOCKS(botThreadCount, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
                    botThreadCount, batchSize, n, nOutputCols, nOutputRows, nOutputPlane,
                    max_displacement, x_shift, neighborhood_grid_width_,
                    kernel_radius_, stride1, stride2, nInputCols, nInputRows,
                    paddedbottomwidth, paddedbottomheight, nInputPlane, inputCount, pad_size,
                    rbot1, gradInput2, gradOutput
                    );
        }

    } else {

        for ( int n = 0; n < batchSize; n++ ) {

            //Bottom0
            CorrelateDataBackward0Subtract_1d<<<GET_BLOCKS(botThreadCount, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>> (
                    botThreadCount,
                    batchSize, n, nOutputCols, nOutputRows, nOutputPlane,
                    max_displacement, x_shift, neighborhood_grid_width_,
                    kernel_radius_, stride1, stride2, nInputCols, nInputRows,
                    paddedbottomwidth, paddedbottomheight, nInputPlane, inputCount, pad_size,
                    gradInput1, rbot1, rbot2, gradOutput
            );
        }

        for (int n = 0; n < batchSize; n++ ) {

            //Bottom0
            CorrelateDataBackward1Subtract_1d<<<GET_BLOCKS(botThreadCount, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
                    botThreadCount,
                    batchSize, n, nOutputCols, nOutputRows, nOutputPlane,
                    max_displacement, x_shift, neighborhood_grid_width_,
                    kernel_radius_, stride1, stride2, nInputCols, nInputRows,
                    paddedbottomwidth, paddedbottomheight, nInputPlane, inputCount, pad_size,
                    rbot1, rbot2, gradInput2, gradOutput
                    );
        }
    }
}

