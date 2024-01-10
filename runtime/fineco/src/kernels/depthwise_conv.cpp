#ifndef DEPTHWISECONV_H
#define DEPTHWISECONV_H


#include "backend/device_api.h"
#include "../backend/cuda/cuda_device_api.cc"
#include "kernel/common.h"
#include "../backend/runtime.cpp"

#include "kernel/layer.h"
#include "kernel/conv.h"

class DepthwiseConv : public Conv2D {
 public:
  DepthwiseConv(size_t N, size_t CI, size_t H, size_t W, size_t KH, size_t KW, size_t OH, size_t OW, int channel_multiplier = 1,RUNTIME runtime = CUDA, DLDataType data_type = DLFLOAT) : 
                               Conv2D(N, CI, H, W, CI, KH, KW, OH, OW, runtime, data_type) {
                                _layer_kind = DEPTHWISECONV;
                                _channel_multiplier = channel_multiplier;
                               }
  

  void InitParams(bool need_input) {
    // TODO, 当前固定死input的输入，不会随机改变，计算完成后Input内存也不释放
    _need_input = need_input;
    if (need_input)
      GenInput();
    GenWeight();
    GenOutput();

  }

  void GenWeight() {
    size_t size = _channel_multiplier * _CI * _KH * _KW;
    _d_weight = GenGPUData(size);
  }

  void Compute(string& candidate_name, DLStream stream) {
    auto device_api =  GetBackendHandle(_runtime);
    auto function = _function_list[candidate_name];
    auto Block = _candidate_list[candidate_name][0];
    auto Thread = _candidate_list[candidate_name][1];
    void *args[4] = { &_d_input, &_d_weight, &_d_output};
    device_api->LaunchKernel(function, Block, Thread, stream, args);
  }

  unsigned long int GetComputation() {
    return _CI * _KH * _KW * _N * _OH * _OW;
  }


 private:
  int _channel_multiplier;
   
};

#endif