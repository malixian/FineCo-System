#ifndef CONVRELU_H
#define CONVRELU_H


#include "backend/device_api.h"
#include "../backend/cuda/cuda_device_api.cc"
#include "kernel/common.h"
#include "../backend/runtime.cpp"

#include "kernel/layer.h"
#include "kernel/conv.h"

class Conv2DRelu : public Conv2D {
 public:
  Conv2DRelu(size_t N, size_t CI, size_t H, size_t W, 
                              size_t CO, size_t KH, size_t KW, size_t OH, size_t OW, RUNTIME runtime = CUDA, DLDataType data_type = DLFLOAT) : 
                               Conv2D(N, CI, H, W, CO, KH, KW, OH, OW, runtime, data_type) {
                                _layer_kind = CONV_RELU;
                               }

  ~Conv2DRelu() {
    auto device_api =  GetBackendHandle(_runtime);
    device_api->FreeDataSpace(0, _d_bias);
  }
  
  void GenBias() {
    size_t size = _N * _CO;
    _d_bias = GenGPUData(size);
  }

  void InitParams(bool need_input) {
    // TODO, 当前固定死input的输入，不会随机改变，计算完成后Input内存也不释放
    _need_input = need_input;
    if (need_input)
      GenInput();
    GenWeight();
    GenBias();
    GenOutput();

  }

  void Compute(string& candidate_name, DLStream stream) {
    auto device_api =  GetBackendHandle(_runtime);
    auto function = _function_list[candidate_name];
    auto Block = _candidate_list[candidate_name][0];
    auto Thread = _candidate_list[candidate_name][1];
    void *args[4] = { &_d_input, &_d_weight, &_d_output, &_d_bias };
    device_api->LaunchKernel(function, Block, Thread, stream, args);
  }


 private:
  float* _d_bias;
   
};

#endif