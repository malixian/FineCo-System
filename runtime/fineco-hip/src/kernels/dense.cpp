#ifndef DENSE_H
#define DENSE_H


#include "backend/device_api.h"
#include "../backend/rocm/rocm_device_api.cc"
#include "kernel/common.h"
#include "../backend/runtime.cpp"

#include "kernel/layer.h"

class Dense : public Layer {
 public:
  Dense(size_t In, size_t Out, RUNTIME runtime = ROCM, DLDataType data_type = DLFLOAT) : _In(In), _Out(Out), Layer(runtime), _data_type(data_type) {
                                _layer_kind = DENSE;
                               }

  ~Dense() {
    auto device_api =  GetBackendHandle(_runtime);
    //device_api->FreeDataSpace(0, _d_input);
    device_api->FreeDataSpace(0, _d_output);
    device_api->FreeDataSpace(0, _d_weight);
    device_api->FreeDataSpace(0, _d_bias);
  }

  void GenWeight() {
    size_t size = _In * _Out;
    _d_weight = GenGPUData(size);
  }

  void GenInput() {
    size_t size = _In;
    _d_input = GenGPUData(size);
  }

  void GenInputAsync(DLStream stream) {
    size_t size = _In;
    _d_input = GenGPUDataAsync(size, stream);
  }


  void GenBias() {
    size_t size = _Out;
    _d_bias = GenGPUData(size);
  }

  void GenOutput() {
    auto device_api =  GetBackendHandle(_runtime);
    size_t size = _Out;
    _d_output = static_cast<float*>(device_api->AllocDataSpace(size, DLFLOAT));
  }

  void InitParams() {
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

  float* GetOutPut() {
    return _d_output;
  }

  void SetInput(float* data) {
    _d_input = data;
  }

  bool hasInput() {
    return (_d_input != nullptr) ? true : false;
  }

  
  string GetInputTensorDim() {
    string result = to_string(_In) ;
    return result;
  }

  string GetOutputTensorDim() {
    string result = to_string(_Out);
    return result;
  }


  unsigned long int GetComputation() {
    return _In * _Out;
  }

 private:
  size_t _In;
  size_t _Out;

  DLDataType _data_type;

  float* _d_weight;
  float* _d_bias;

 public:
   float* _d_output;
   float* _d_input = nullptr;
   
};

#endif