#ifndef SLEEP_H
#define SLEEP_H

#include "common.h"
#include "../src/backend/runtime.cpp"

class Sleep {
 public:

  virtual void Compute(DLStream stream, int ius) {
    auto device_api =  GetBackendHandle(ROCM);
    auto Block = 1;
    auto Thread = 1;
    int* us = &ius;
    void *args[1] = {us};
    device_api->LaunchKernel(_function, Block, Thread, stream, args);
  }

  void Load() {
    string candidate_name = "sleep";
    int Block = 1;
    int Thread = 1;
    auto device_api = GetBackendHandle(ROCM);
    device_api->CompileKernelByPath("", candidate_name);
    
    auto function = static_cast<hipFunction_t>(device_api->GetFunctionByPath("",   candidate_name));
    _function = function;
    
  }

 private:
  hipFunction_t _function;
   
};

#endif