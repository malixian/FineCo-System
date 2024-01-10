#ifndef ROCM_DEVICE_API_H
#define ROCM_DEVICE_API_H

#include "backend/device_api.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>


#define HIP_CALL(ret)                                       \
  {                                                         \
    hipError_t e = ret;                                     \
    if (e != hipSuccess) {                                  \
        cout << "HIP Error: " << e << endl; exit(-1);}      \
  }



inline bool exists_file(const std::string& name) {
  struct stat buffer;   
  return (stat(name.c_str(), &buffer) == 0); 
}

class ROCMDeviceAPI final : public DeviceAPI {

private:

    ROCMDeviceAPI() {
        //cuInit(0);
        
        //cuDeviceGet(&_device, 0) ;
        //cuCtxCreate(&_context, 0, _device); 
        
        /* UNSUPPORTED_EXEC_AFFINITY
        CUexecAffinityParam affinity;
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinity.param.smCount.val = 20;
        CUDA_DRIVER_CALL(cuCtxCreate_v3(&_context, &affinity, 1, 0, _device));
        */
        //cout<<"Init CUDA Device API"<<endl;
        //CUDA_CALL(cudaSetDevice(0));
    }

    ~ROCMDeviceAPI() {
        //CUDA_CALL(cudaDeviceReset());
    }
 
public:

    void SetDevice(int dev) final { HIP_CALL(hipSetDevice(dev)); }
  
    void* AllocDataSpace(size_t nbytes, DLDataType type_hint) final {
        float* ret;
        size_t aloc_size = 0;
        if (type_hint == DLFLOAT)
            aloc_size = nbytes * sizeof(float);
        else if (type_hint == DLINT)
            aloc_size = nbytes * sizeof(int);
        else
            aloc_size = nbytes * sizeof(float);
        HIP_CALL(hipMalloc((void**)&ret, aloc_size));
        return ret;
    }

    void FreeDataSpace(int dev, void* ptr) final {
        // HIP_CALL(hipSetDevice(dev));
        HIP_CALL(hipFree(ptr));
    }

    DLStream CreateStream(int dev) final {
        // HIP_CALL(hipSetDevice(dev));
        hipStream_t retval;
        
        const char *cumask_on = "ENABLE_CUMASK";
        const char *mask[4] = {"CUMASK_0", "CUMASK_1", "CUMASK_2", "CUMASK_3"};
        uint32_t cu_mask[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0x00ffffff};

        char *tmp = getenv(cumask_on);
        if (tmp && (atoi(tmp) > 0)) {
            for (int i = 0; i < 4; i++) {
                tmp = getenv(mask[i]);
                if (tmp)
                    sscanf(tmp, "%x", &cu_mask[i]);
                // printf("%x\n", cu_mask[i]);
            }
            HIP_CALL(hipExtStreamCreateWithCUMask(&retval, 4, cu_mask));
            // printf("ENABLE CU MASK!\n");
        } else
            HIP_CALL(hipStreamCreate(&retval));
            
        // HIP_CALL(hipStreamCreate(&retval));
        return static_cast<DLStream>(retval);
    }

    DLStream CreateStream(int cuMaskSize, uint32_t *cuMask) final {
        // HIP_CALL(hipSetDevice(dev));
        hipStream_t retval;
        HIP_CALL(hipExtStreamCreateWithCUMask(&retval, cuMaskSize, cuMask));
        return static_cast<DLStream>(retval);
    }

    DLStream CreateStreamWithPriority(int priority, int dev) final {
        hipStream_t stream;
        HIP_CALL(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority));
        return static_cast<DLStream>(stream);
    }

    void FreeStream(int dev, DLStream stream) final {
        // HIP_CALL(hipSetDevice(dev));
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        HIP_CALL(hipStreamDestroy(hip_stream));
    }


    void StreamSync(DLStream stream, int dev=0) final {
        // HIP_CALL(hipSetDevice(dev));
        HIP_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
    }

    void CopyDataFromTo(void* from, void* to, size_t size, MemCpyDirct kind) final {
        if (kind == D2H)
            HIP_CALL(hipMemcpy(to, from, size, hipMemcpyDeviceToHost));
        if (kind == H2D)
            HIP_CALL(hipMemcpy(to, from, size, hipMemcpyHostToDevice));
    }

    void CopyDataFromToAsync(void* from, void* to, size_t size, MemCpyDirct kind, DLStream stream) final {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        if (kind == D2H)
            HIP_CALL(hipMemcpyAsync(to, from, size, hipMemcpyDeviceToHost, hip_stream));
        if (kind == H2D)
            HIP_CALL(hipMemcpyAsync(to, from, size, hipMemcpyHostToDevice, hip_stream));
    }

    void CompileKernel(
        const string& model_name, 
        const string& layer_idx_str, 
        const string& kernel_name, 
        const string& layer_name = "") {
        string path = model_name + "/";
        if (layer_idx_str != "")
            path += layer_idx_str + "/";
         path += layer_name;
        CompileKernelByPath(path, kernel_name);

        // string path = "/home/chr/repos/fineco-hip/include/kernel/cuda/" + model_name + "/";     // ! include/kernel/cuda/ 路径不知道还要不要改
        // if (layer_idx_str != "")
        //     path += layer_idx_str + "/";
        // path += layer_name;

        // cout << "compile path: " << path << endl;
        // string cd_path = "cd " + path;
        // string file_name = kernel_name + ".cu";
        // string cmd = cd_path + " && " "nvcc -arch=sm_80 --cubin " + file_name;
        // string output = path + "/" + kernel_name + ".cubin";
        // if (exists_file(output)) {
        //     //cout<< layer_idx_str + " " + kernel_name + " Has Exist"<<endl;
        //     return;
        // }

        // system(cmd.c_str());
        // if (exists_file(output)) {
        //     std::cout<< layer_idx_str + " " + kernel_name + " Compile Successfully "<<std::endl;
        // } else {
        //     std::cout<<"Kernel Compile Failed "<<std::endl;
        // }
    }

    void CompileKernelByPath(const string& path, const string& kernel_name) {
        string base_path = "/home/chr/repos/fineco-hip/include/kernel/rocm/";

        if (path != "") {
            base_path +=  path + "/";
        }
        
        // cout << "compile path: " << path << endl;
        string cd_path = "cd " + base_path;
        string file_name = kernel_name + ".cu";
        string cmd = cd_path + " && " "hipcc --genco " + file_name + " -o " + kernel_name + ".hsaco";
        string output = base_path + kernel_name + ".hsaco";
        if (exists_file(output)) {
            //cout<< layer_idx_str + " " + kernel_name + " Has Exist"<<endl;
            return;
        }

        system(cmd.c_str());
        if (exists_file(output)) {
            // std::cout<<  base_path + " " + kernel_name + " Compile Successfully "<<std::endl;
        } else {
            std::cout << "Kernel Compile Failed " << std::endl;
            std::cout << "  path: " + base_path + " " + file_name << std::endl;
        }
    }



    void *GetFunction(const string& model_name, const string& layer_idx_str, const string& kernel_name, const string& layer_name="") {
        string path = model_name + "/";
        if (layer_idx_str != "")
            path += layer_idx_str + "/";
        path += layer_name;
        return GetFunctionByPath(path, kernel_name);

        // hipModule_t module;
        // hipFunction_t function;

        // string path = "/home/chr/repos/fineco-hip/include/kernel/cuda/" + model_name + "/";
        // if (layer_idx_str != "")
        //     path += layer_idx_str + "/";
        // path += layer_name;
        
        // string module_file = path + "/" + kernel_name + ".cubin";
        // //cout<<"get function path: "<<module_file<<endl;
        // HIP_CALL(hipModuleLoad(&module, module_file.c_str()));
        // HIP_CALL(hipModuleGetFunction(&function, module, kernel_name.c_str()));
        // return function;
    }

    void *GetFunctionByPath(const string& path, const string& kernel_name) {
        hipModule_t module;
        hipFunction_t function;

        string base_path = "/home/chr/repos/fineco-hip/include/kernel/rocm/";
        if (path != "")
            base_path += path + "/";
        string module_file = base_path + kernel_name + ".hsaco";
        // cout << "get function path: " << module_file << endl;
        HIP_CALL(hipModuleLoad(&module, module_file.c_str()));
        HIP_CALL(hipModuleGetFunction(&function, module, "default_function_kernel0"));
        return function;
    }

    void LaunchKernel (void* function, size_t grid, size_t block, DLStream stream, void** args) {
        // hipCtxSetCurrent(_context);
        // cout << Grid << " " << Block << endl;
        hipFunction_t f = static_cast<hipFunction_t>(function);
        hipStream_t strm = static_cast<hipStream_t>(stream);

        HIP_CALL(hipModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, strm, args, 0));
        // dim3 g = {(uint32_t)grid, 1, 1}, b = {(uint32_t)block, 1, 1};
        // HIP_CALL(hipLaunchKernelGGL(f, g, b, 0, strm, args));

        //auto ret = cuLaunchKernel(cu_function, Grid, 1, 1, Block, 1, 1, 0, custream, args, 0);
        //cout<< "launch kernel result:"<<ret<<endl;
    }
    
    void DeviceSync() {
        HIP_CALL(hipDeviceSynchronize());
    }

    static ROCMDeviceAPI* GetROCMDeviceAPI() {
        if (_inst == NULL) {
            _inst = new ROCMDeviceAPI();
        } 
        return _inst;
    }

private:
    static ROCMDeviceAPI* _inst;
    hipDevice_t _device;
    hipCtx_t _context;

};

// init static member
ROCMDeviceAPI* ROCMDeviceAPI::_inst = NULL;

#endif