#if 0
#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include "cuda_device_api.cc"
#include "backend/timer.h"

class CUDATimer : public Timer {
 public:
  CUDATimer(DLStream stream) {
    stream_ = static_cast<CUstream>(stream);
    CUDA_CALL(cudaEventCreate(&start_));
    CUDA_CALL(cudaEventCreate(&stop_));
  }

  void Start() {
    CUDA_CALL(cudaEventRecord(start_, stream_));
  }

  void Stop() {
    CUDA_CALL(cudaEventRecord(stop_, stream_)); 
  }

  float SyncAndGetElapsedms() {
    CUDA_CALL(cudaEventSynchronize(stop_));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

  ~CUDATimer() {
    CUDA_CALL(cudaEventDestroy(start_));
    CUDA_CALL(cudaEventDestroy(stop_));
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  CUstream stream_;

};

#endif
#endif