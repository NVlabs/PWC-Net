#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/warp_layer.hpp"

// by deqings@nvidia.com Copyright (C) 2018 NVIDIA Corporation. All rights reserved. 

namespace caffe {

template <typename Dtype>
void WarpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
    // string prefix = "\t\tWarp Layer:: Reshape: \t";
    // std::cout<<"Getting output_H_ and output_W_"<<std::endl;

    output_H_ = bottom[0]->shape(2);
    output_W_ = bottom[0]->shape(3);

    // std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

}

template <typename Dtype>
void WarpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
    string prefix = "\t\tWarp Layer:: Reshape: \t";

    //if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

    // image/features
    N = bottom[0]->shape(0);
    C = bottom[0]->shape(1);
    H = bottom[0]->shape(2);
    W = bottom[0]->shape(3);

    // reshape V
    vector<int> shape(4);

    shape[0] = N;
    shape[1] = C;
    shape[2] = output_H_;
    shape[3] = output_W_;

    top[0]->Reshape(shape);

    if (bottom[1]->shape(1) != 2)
      std::cout<<"error: flow field should have two channels!\n";
}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //NOT_IMPLEMENTED;
  CHECK(1) << "Only support GPU now";
}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //NOT_IMPLEMENTED;
  CHECK(1) << "Only support GPU now";
}

#ifdef CPU_ONLY
STUB_GPU(WarpLayer);
#endif

INSTANTIATE_CLASS(WarpLayer);
REGISTER_LAYER_CLASS(Warp);

}  // namespace caffe
