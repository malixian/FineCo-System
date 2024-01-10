#ifndef ADD_H
#define ADD_H

#include "common.h"
#include "../src/backend/runtime.cpp"
#include "layer.h"

class TensorAdd : public Layer {
 public:
  TensorAdd( size_t M, size_t N, size_t K,  RUNTIME runtime = CUDA, DLDataType data_type = DLFLOAT) : 
                                _M(M), _N(N), _K(K), Layer(runtime), _data_type(data_type) {
                                _layer_kind = TENSORADD;
  }
  
  void GenInput1() {
    size_t size = _N * _M * _K ;
    _d_input_1 = GenGPUData(size);
  }

  void GenInput2() {
    size_t size = _N * _M * _K ;
    _d_input_2 = GenGPUData(size);
  }

  void GenOutput() {
    auto device_api =  GetBackendHandle(_runtime);
    size_t size = _N * _M * _K;
    _d_output = static_cast<float*>(device_api->AllocDataSpace(size, DLFLOAT));
  }

  virtual void InitParams(bool need_input) {
    // TODO, 当前固定死input的输入，不会随机改变，计算完成后Input内存也不释放
    if (need_input) {
      GenInput1();
      GenInput2();
    }
    GenOutput();
  }

  virtual void Compute(string& candidate_name, DLStream stream) {
    auto device_api =  GetBackendHandle(_runtime);
    auto function = _function_list[candidate_name];
    auto Block = _candidate_list[candidate_name][0];
    auto Thread = _candidate_list[candidate_name][1];
    void *args[3] = { &_d_input_1, &_d_input_2, &_d_output};
    device_api->LaunchKernel(function, Block, Thread, stream, args);
  }


  float* GetOutPut() {
    return _d_output;
  }

  void SetInput1(float* data) {
    _d_input_1 = data;
  }

  void SetInput2(float* data) {
    _d_input_2 = data;
  }

  void SetPhaseName(BERTPhase phase_name) {
    _phase_name = phase_name;
  }

  BERTPhase GetPhaseName() {
    return _phase_name;
  }

 protected:
  size_t _N;
  size_t _M;
  size_t _K;

  DLDataType _data_type;
  
 public:
   float* _d_output;
   float* _d_input_1;
   float* _d_input_2;
   BERTPhase _phase_name;
   
};

#endif