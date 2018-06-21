#ifndef WARP_LAYER_HPP_
#define WARP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

// by deqings@nvidia.com Copyright (C) 2018 NVIDIA Corporation. All rights reserved. 

namespace caffe {

/**
 * @brief Warp the H*W*nchannels*nbatch image/feature using the H*W*2 optical flow
 *    the first layer bottom[0] contains the image/features
 *    the second layer bottom[1] contains the flow field
 */
template <typename Dtype>
class WarpLayer : public Layer<Dtype> {
 public:

  explicit WarpLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Warp"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // int num_;
  // int top_height_, top_width_;
  // int top_channels_;
  
  int N, C, H, W;
  int output_H_, output_W_;

  // Warp specific:
  
};


}  // namespace caffe

#endif  // Warp_LAYER_HPP_
