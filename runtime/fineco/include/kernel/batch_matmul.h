#ifndef BATCH_MATMUL_H
#define BATCH_MATMUL_H

#include "common.h"
#include "../src/backend/runtime.cpp"
#include "layer.h"

class BatchMatmul : public Layer {
 public:
  BatchMatmul(size_t B, size_t M, size_t N, size_t K,  RUNTIME runtime = CUDA, DLDataType data_type = DLFLOAT) : 
                               _B(B), _M(M), _N(N), _K(K), Layer(runtime), _data_type(data_type) {
                                _layer_kind = BATCH_MATMUL;
  }

  ~BatchMatmul() {
    auto device_api =  GetBackendHandle(_runtime);
    
    
    if(_need_input_1)
      device_api->FreeDataSpace(0, _d_input_1);
    
    if(_need_input_2)
      device_api->FreeDataSpace(0, _d_input_2);
    
    device_api->FreeDataSpace(0, _d_output);
    
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

  virtual void InitParams(bool need_input_1, bool need_input_2) {
    // TODO, 当前固定死input的输入，不会随机改变，计算完成后Input内存也不释放
    if (need_input_1) {
      _need_input_1 = true;
      GenInput1();
    }
    if (need_input_2) {
      _need_input_2 = true;
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

  float* GetInput1() {
    return _d_input_1;
  }

  float* GetInput2() {
    return _d_input_2;
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

  unsigned long int GetComputation() {
    return _B * _N * _M * _K;
  }

 private:
  bool _need_input_1 = false;
  bool _need_input_2 = false;

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
   BERTPhase _phase_name;

   
};

#endif