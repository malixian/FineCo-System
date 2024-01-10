#ifndef RUNTIME_H
#define RUNTIME_H

#include "backend/device_api.h"
#include "rocm/rocm_device_api.cc"

enum RUNTIME {
    CUDA,
    ROCM,
};

DeviceAPI* GetBackendHandle(RUNTIME runtime) {
    if (runtime == ROCM) {
        return ROCMDeviceAPI::GetROCMDeviceAPI();
    } else {
        return nullptr;
    }
}

#endif