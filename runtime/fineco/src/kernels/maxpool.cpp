#ifndef MAXPOOL_H
#define MAXPOOL_H


#include "backend/device_api.h"
#include "../backend/cuda/cuda_device_api.cc"
#include "kernel/common.h"
#include "../backend/runtime.cpp"

#include "kernel/layer.h"

class MaxPool : public Layer {
 public:
  MaxPool(size_t N, size_t C, size_t OH, size_t OW, 
            RUNTIME runtime = CUDA, DLDataType data_type = DLFLOAT) : 
            _N(N), _C(C), _OH(OH), _OW(OW), Layer(runtime), _data_type(data_type) {
                _layer_kind = MAXPOOL;
            }
  
  ~MaxPool() {
    auto device_api =  GetBackendHandle(_runtime);
    //device_api->FreeDataSpace(0, _d_input);
    device_api->FreeDataSpace(0, _d_output);
  }

  void GenOutput() {
    auto device_api =  GetBackendHandle(_runtime);
    size_t size = _N * _C * _OH * _OW;
    _d_output = static_cast<float*>(device_api->AllocDataSpace(size, DLFLOAT));
  }

  void InitParams() {
    GenOutput();
  }

  void Compute(string& candidate_name, DLStream stream) {
    auto device_api =  GetBackendHandle(_runtime);
    auto function = _function_list[candidate_name];
    auto Block = _candidate_list[candidate_name][0];
    auto Thread = _candidate_list[candidate_name][1];
    void *args[4] = { &_d_output, &_d_input};
    device_api->LaunchKernel(function, Block, Thread, stream, args);
  }

  float* GetOutPut() {
    return _d_output;
  }

  void SetInput(float* data) {
    _d_input = data;
  }

  float* GetHostOutput(DLStream stream) {
    size_t mem_size =  sizeof(float) * _N * _C * _OH * _OW;
    float* h = (float*)malloc(mem_size);
    
    GetBackendHandle(_runtime)->CopyDataFromToAsync(_d_output, h, mem_size, D2H, stream);
    return h;
  }

  int GetOutPutSize() {
    return _N * _C * _OH * _OW;
  }

  bool hasInput() {
    return (_d_input != nullptr) ? true : false;
  }


 private:
  size_t _N;
  size_t _C;
  size_t _OH;
  size_t _OW;

  DLDataType _data_type;

 public:
   float* _d_output;
   float* _d_input;
   
   
};

#endif