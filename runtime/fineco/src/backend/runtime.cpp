#ifndef RUNTIME_H
#define RUNTIME_H

#include "backend/device_api.h"
#include "cuda/cuda_device_api.cc"

enum RUNTIME {
    CUDA,
    ROCM,
};

DeviceAPI* GetBackendHandle(RUNTIME runtime) {
    if (runtime == CUDA) {
        return CUDADeviceAPI::GetCUDADeviceAPI();
    } else {
        return nullptr;
    }
}

#endif