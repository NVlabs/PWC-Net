#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/warp_layer.hpp"
#include "caffe/util/benchmark.hpp"

// by deqings@nvidia.com Copyright (C) 2018 NVIDIA Corporation. All rights reserved. 

namespace caffe {

const int num_flow_channels = 2; // channels of flow are always 2

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, Dtype value, int size, 
  int i, Dtype* dst) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    dst[index * size + i] = value;
  }
}

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
  const Dtype* src, int size_dst, int i, Dtype* dst) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    dst[index * size_dst + i] = src[index * size_src + k];
  }
}

template <typename Dtype>
__global__ void WarpForwardGPU(const int nthreads, int N, int C,
    int output_H_, int output_W_, int H, int W,
    const Dtype* flow, const Dtype* U, Dtype* V) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {

    const int t = index % output_W_; // horizontal index x width
    const int s = (index / output_W_) % output_H_; // vertical index y height
    const int j = (index / (output_W_ * output_H_)) % C; // channel
    const int i = index / (output_W_ * output_H_ * C);   // batch number

    const int V_offset = index;
    V[V_offset] = (Dtype)0.;

    // interpolation position
    const int vflow_offset = output_W_*output_H_;    
    const int flow_offset  = i * (num_flow_channels * output_W_*output_H_);    
    const Dtype x =  t + flow[s*output_W_ + t + flow_offset ];                      // add horizontal flow to horizontal (x) coordinate
    const Dtype y =  s + flow[s*output_W_ + t + vflow_offset + flow_offset]; // add vertical flow to vertical (y) coordinate    

    if (x >=0 && x<= W-1 && y >=0 && y<=H-1) {
      // interpolation position is inside image boundary

      //int m, n; 
      Dtype w;
      const Dtype* pic = U + i * (C * H * W) + j * (H * W);

      int fx, fy, cx, cy;
      fx = floor(x); 
      fy = floor(y);
      cx = ceil(x);
      cy = ceil(y);
      Dtype alpha_x = x-fx;
      Dtype alpha_y = y-fy;

      // top left neighbor
      w = (1 - alpha_x) * (1 - alpha_y);
      V[V_offset] += w * pic[fy * W + fx];
      
      // top rigth neighbor
      w = alpha_x * (1 - alpha_y);
      V[V_offset] += w * pic[fy * W + cx];

      // bottom left neighbor
      w = (1 - alpha_x) * alpha_y;
      V[V_offset] += w * pic[cy * W + fx];
      
      // bottom right neighbor
      w = alpha_x * alpha_y;
      V[V_offset] += w * pic[cy * W + cx];      

    }
  }
}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  string prefix = "WarpLayer::Forward_gpu::\t";

  const Dtype* U = bottom[0]->gpu_data();
  const Dtype* flow = bottom[1]->gpu_data();

  Dtype* V = top[0]->mutable_gpu_data();

  caffe_gpu_set(top[0]->count(), (Dtype)0, V);
  
  const int nthreads = N * C * output_H_ * output_W_;

  WarpForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, flow, U, V);
}

//backpropagate gradient w.r.t. the input flow field
template <typename Dtype>
__global__ void WarpBackwardGPU_dFlow(const int nthreads, int C,
    int output_H_, int output_W_, int H, int W,
    const Dtype* flow, const Dtype* dV, const Dtype* U,  
    Dtype* dFlow) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {

    const int t = index % output_W_;                      // horizontal index x width
    const int s = (index / output_W_) % output_H_;        // vertical index y height
    const int j = (index / (output_W_ * output_H_)) % C;  // channel
    const int i = index / (output_W_ * output_H_ * C);    // batch number

    // interpolation position
    const int vflow_offset = output_W_*output_H_;    
    const int flow_offset  = i * (num_flow_channels * output_W_*output_H_); 
    

    const Dtype x =  t + flow[s*output_W_ + t + flow_offset];                // add horizontal flow to horizontal (x) coordinate
    const Dtype y =  s + flow[s*output_W_ + t + vflow_offset + flow_offset]; // add vertical flow to vertical (y) coordinate    

    // Already initialized, should not initialize gradient w.r.t horizontal and vertical flow field again
    // the gradients need to be accumulated across image channel 
    // dFlow[flow_offset + s*output_W_ + t] = (Dtype)0.;               // horizontal
    // dFlow[flow_offset + s*output_W_ + t + vflow_offset] = (Dtype)0.; // vertical     

    if (x >=0 && x<= W-1 && y >=0 && y<=H-1) {
      // interpolation position is inside image boundary

      // pointer to image/feature data
      const Dtype* pic = U + i * (C * H * W) + j * (H * W);

      int fx, fy, cx, cy;
      fx = floor(x); 
      fy = floor(y);
      cx = ceil(x);
      cy = ceil(y);
      Dtype alpha_x = x-fx;
      Dtype alpha_y = y-fy;

      // need atomic addition because different image channels will write to the same position in parallel
      // w.r.t horizontal flow 
      caffe_gpu_atomic_add(  ((1-alpha_y)*pic[fy * W + cx]-(1-alpha_y)*pic[fy * W + fx] +alpha_y*pic[cy * W + cx]-alpha_y*pic[cy * W + fx])*dV[index], dFlow + flow_offset + s*output_W_ + t);
      // w.r.t vertical flow 
      caffe_gpu_atomic_add( ((1-alpha_x)*pic[cy * W + fx]-(1-alpha_x)*pic[fy * W + fx] +alpha_x*pic[cy * W + cx]-alpha_x*pic[fy * W + cx])*dV[index], dFlow + flow_offset + s*output_W_ + t + vflow_offset);

      // integer grid processing
      if (fx==cx){        
        // same as numerical approximation: average of gradient 
        Dtype top_left, bottom_left, top_right, bottom_right;
        int fxn = fx-1;
        int fxp = fx+1;
        // there may be inconsistency for out-of-boundary pixels: caffe's numerical scheme should not be used          
        if (fxn<0){
          top_left =.0;
          bottom_left =.0;
        }          
        else {
          top_left    = pic[fy * W + fx] - pic[fy * W + fxn];
          bottom_left = pic[cy * W + fx] - pic[cy * W + fxn];;
        }
        if (fxp>W-1){
          top_right =.0;
          bottom_right =.0;
        }          
        else {
          top_right    = pic[fy * W + fxp] - pic[fy * W + fx];
          bottom_right = pic[cy * W + fxp] - pic[cy * W + fx];
        }
        caffe_gpu_atomic_add( Dtype( (1-alpha_y) *(top_right+top_left)/2. + alpha_y*(bottom_right+bottom_left)/2.)*dV[index], dFlow + flow_offset + s*output_W_ + t);
      }

      if (fy==cy){
        // same as numerical approximation: average of gradient 
        Dtype top_left, bottom_left, top_right, bottom_right;
        int fyn = fy-1;
        int fyp = fy+1;
        if (fyn<0){
          top_left  =.0;
          top_right =.0;          
        }          
        else {
          top_left    = pic[fy * W + fx] - pic[fyn * W + fx];
          top_right   = pic[fy * W + cx] - pic[fyn * W + cx];
        }
        if (fyp>H-1){
          bottom_left  =.0;
          bottom_right =.0;
        }          
        else {
          bottom_left  = pic[fyp * W + fx] - pic[fy * W + fx];
          bottom_right = pic[fyp * W + cx] - pic[fy * W + cx];
        }
        caffe_gpu_atomic_add( Dtype( (1-alpha_x) *(bottom_left+top_left)/2. + alpha_x*(bottom_right+top_right)/2.)*dV[index], dFlow + flow_offset + s*output_W_ + t + vflow_offset);
      }

      
    }
    
  }
}

//backpropagate gradient w.r.t. the input image/feature
template <typename Dtype>
__global__ void WarpBackwardGPU_dU(const int nthreads, const int C, 
  const int W,  const int H, const int output_H_, const int output_W_, 
  const Dtype* flow, const Dtype* dV, Dtype* dU) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {

    const int t = index % output_W_;                      // horizontal index x width
    const int s = (index / output_W_) % output_H_;        // vertical index y height
    const int j = (index / (output_W_ * output_H_)) % C;  // channel
    const int i = index / (output_W_ * output_H_ * C);    // batch number

   // interpolation position
    const int vflow_offset = output_W_*output_H_;    
    const int flow_offset  = i * (num_flow_channels * output_W_*output_H_);    
    const Dtype x =  t + flow[s*output_W_ + t + flow_offset ];               // add horizontal flow to horizontal (x) coordinate
    const Dtype y =  s + flow[s*output_W_ + t + flow_offset + vflow_offset]; // add vertical flow to vertical (y) coordinate    

    if (x >=0 && x<= W-1 && y >=0 && y<=H-1) {

      // interpolation position is inside image boundary
      // int m, n; 
      Dtype w;
      const int dU_offset = i * (C * output_W_ * output_H_) + j * (output_W_ * output_H_);

      int fx, fy, cx, cy;
      fx = floor(x); 
      fy = floor(y);
      cx = ceil(x);
      cy = ceil(y);
      Dtype alpha_x = Dtype(x-fx);
      Dtype alpha_y = Dtype(y-fy);

      // top left neighbor
      w = (1 - alpha_x) * (1 - alpha_y);
      caffe_gpu_atomic_add(w * dV[index], dU + dU_offset + fy * W + fx );
      
      // top right neighbor
      w = alpha_x * (1 - alpha_y);
      caffe_gpu_atomic_add(w * dV[index], dU + dU_offset + fy * W + cx );

      // bottom left neighbor
      w = (1 - alpha_x) * alpha_y;
      caffe_gpu_atomic_add(w * dV[index], dU + dU_offset + cy * W + fx );

      // bottom right neighbor
      w = alpha_x * alpha_y;
      caffe_gpu_atomic_add(w * dV[index], dU + dU_offset + cy * W + cx );

    }
  }
}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  string prefix = "WarpLayer::Backward_GPU::\t";

  const Dtype* dV   = top[0]->gpu_diff();
  const Dtype* U    = bottom[0]->gpu_data();
  const Dtype* flow = bottom[1]->gpu_data();  

  const int nthreads = N * C * output_H_ * output_W_;

  // derivative w.r.t the flow field
  Dtype* dFlow = bottom[1]->mutable_gpu_diff();
  caffe_gpu_set(bottom[1]->count(), (Dtype)0., dFlow);
  WarpBackwardGPU_dFlow<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, flow,
          dV, U, dFlow);

  // derivative w.r.t the features
  Dtype* dU = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
  WarpBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, W, H, output_H_, output_W_, flow, dV, dU);
  
}

INSTANTIATE_LAYER_GPU_FUNCS(WarpLayer);

} // namespace caffe