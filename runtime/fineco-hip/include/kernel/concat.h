#ifndef CONCAT_H
#define CONCAT_H

#include "common.h"
#include "../src/backend/runtime.cpp"
#include "layer.h"

class Concat : public Layer {
 public:
  Concat(size_t B, size_t M, size_t N, size_t K,  RUNTIME runtime = ROCM, DLDataType data_type = DLFLOAT) : 
                               _B(B), _M(M), _N(N), _K(K), Layer(runtime), _data_type(data_type) {
                                _layer_kind = BATCH_MATMUL;
  }
  
  void GenInput1() {
    size_t size = _B * _M * _K ;
    _d_input_1 = GenGPUData(size);
  }

  void GenInput2() {
    size_t size = _B * _N * _K ;
    _d_input_2 = GenGPUData(size);
  }

  void GenOutput() {
    auto device_api =  GetBackendHandle(_runtime);
    size_t size = _B * _M * _N;
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

 protected:
  size_t _B;
  size_t _N;
  size_t _M;
  size_t _K;

  DLDataType _data_type;
  
 public:
   float* _d_output;
   float* _d_input_1;
   float* _d_input_2;
   
};

#endif