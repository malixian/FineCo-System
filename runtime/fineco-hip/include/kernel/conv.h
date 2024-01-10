#ifndef CONV_H
#define CONV_H

#include "common.h"
#include "../src/backend/runtime.cpp"
#include "layer.h"

class Conv2D : public Layer {
 public:
  Conv2D(size_t N, size_t CI, size_t H, size_t W, 
                              size_t CO, size_t KH, size_t KW, size_t OH, size_t OW, RUNTIME runtime = ROCM, DLDataType data_type = DLFLOAT) : 
                               _N(N), _CI(CI), _H(H), _W(W), _CO(CO), _KH(KH), _KW(KW), _OH(OH), _OW(OW), Layer(runtime), _data_type(data_type) {
                                _layer_kind = CONV;
                               }
  
  ~Conv2D() {
    auto device_api =  GetBackendHandle(_runtime);
    if(_need_input)
      device_api->FreeDataSpace(0, _d_input);
    device_api->FreeDataSpace(0, _d_weight);
    device_api->FreeDataSpace(0, _d_output);
    
  }

  virtual void GenInput() {
    size_t size = _N * _CI * _H * _W;
    _d_input = GenGPUData(size);
  }

  virtual void CopyAndSetInput(float* input, DLStream stream) {
    GetBackendHandle(_runtime)->CopyDataFromToAsync(input, _d_input, _N * _CI * _H * _W, H2D, stream);
  }

  virtual void GenInputAsync(DLStream stream) {
    size_t size = _N * _CI * _H * _W;
    _d_input = GenGPUDataAsync(size, stream);
  }

  virtual void GenWeight() {
    size_t size = _CI * _CO * _KH * _KW;
    _d_weight = GenGPUData(size);
  }

  void GenOutput() {
    auto device_api =  GetBackendHandle(_runtime);
    size_t size = _N * _CO * _OH * _OW;
    _d_output = static_cast<float*>(device_api->AllocDataSpace(size, DLFLOAT));
  }

  virtual void InitParams(bool need_input) {
    // TODO, 当前固定死input的输入，不会随机改变，计算完成后Input内存也不释放
    _need_input = need_input;
    if (need_input)
      GenInput();
    GenWeight();
    GenOutput();
  }

  float* GetHostOutput(DLStream stream) {
    size_t mem_size =  sizeof(float) * _N * _CO * _OH * _OW;
    float* h = (float*)malloc(mem_size);
    
    GetBackendHandle(_runtime)->CopyDataFromToAsync(_d_output, h, mem_size, D2H, stream);
    return h;
  }

  

  virtual void Compute(string& candidate_name, DLStream stream) {
    auto device_api =  GetBackendHandle(_runtime);
    auto function = _function_list[candidate_name];
    auto Block = _candidate_list[candidate_name][0];
    auto Thread = _candidate_list[candidate_name][1];
    void *args[3] = { &_d_input, &_d_weight, &_d_output};
    device_api->LaunchKernel(function, Block, Thread, stream, args);
  }

  uint GetInputTensorSize() {
    return _N * _CI * _H * _W;
  }

  uint GetOutTensorSize() {
    return _N * _CO * _OH * _OW;
  }

  string GetInputTensorDim() {
    string result = to_string(_N) + " " +  to_string(_CI) + " " +  to_string(_H) + " " +  to_string(_W);
    return result;
  }

  string GetOutputTensorDim() {
    string result = to_string(_N) + " " +  to_string(_CO) + " " +  to_string(_OH) + " " +  to_string(_OW);
    return result;
  }


  float* GetOutPut() {
    return _d_output;
  }

  float* GetInput() {
    return _d_input;
  }

  void SetInput(float* data) {
    _d_input = data;
  }

  bool hasInput() {
    return (_d_input != nullptr) ? true : false;
  }

  unsigned long int GetComputation() {
    return _N * _CI * _CO * _KH * _KW * _OH * _OW;
  }
  

 protected:
  size_t _N;
  size_t _CI;
  size_t _H;
  size_t _W;
  size_t _CO;
  size_t _KH;
  size_t _KW;
  size_t _OH;
  size_t _OW;

  DLDataType _data_type;

  float* _d_weight;

  bool _need_input;

 public:
   float* _d_output;
   float* _d_input = nullptr;
   
};

#endif