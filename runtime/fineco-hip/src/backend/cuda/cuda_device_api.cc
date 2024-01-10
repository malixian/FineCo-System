#if 0
#ifndef CUDA_DEVICE_API_H
#define CUDA_DEVICE_API_H

#include "backend/device_api.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <iostream>
#include<stdlib.h>
#include <sys/stat.h>


#define CUDA_CALL(ret)                                       \
  {                                                           \
    cudaError_t e = ret;                                   \
    if (e != cudaSuccess) { \
        cout << "CUDA Error: " << e << "file:"<<__FILE__<<" line:"<<__LINE__<<endl; exit(-1);}                 \
  }

#define CUDA_DRIVER_CALL(ret)                                       \
  {                                                           \
    CUresult e = ret;                                   \
    if (e != CUDA_SUCCESS) { \
        cout << "CUDA DRIVER Error: " << e << "file:"<<__FILE__<<" line:"<<__LINE__<<endl; exit(-1);}                 \
  }




inline bool exists_file (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

class CUDADeviceAPI final : public DeviceAPI {
 private:

  CUDADeviceAPI () {
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

  ~CUDADeviceAPI() {
    //CUDA_CALL(cudaDeviceReset());
  }
 
 public:

  void SetDevice(int dev) final { CUDA_CALL(cudaSetDevice(dev)); }
  
  void* AllocDataSpace(size_t nbytes, DLDataType type_hint) final {
    float* ret;
    size_t aloc_size = 0;
    if (type_hint == DLFLOAT)
      aloc_size = nbytes * sizeof(float);
    else if(type_hint == DLINT)
      aloc_size = nbytes * sizeof(int);
    else
      aloc_size = nbytes * sizeof(float);
    CUDA_CALL(cudaMalloc((void**)&ret, aloc_size));
    return ret;
  }

  void FreeDataSpace(int dev, void* ptr) final {
    //CUDA_CALL(cudaSetDevice(dev));
    CUDA_CALL(cudaFree(ptr));
  }

  DLStream CreateStream(int dev) final {
    //CUDA_CALL(cudaSetDevice(dev));
    cudaStream_t retval;
    CUDA_CALL(cudaStreamCreate(&retval));
    return static_cast<DLStream>(retval);
  }

  DLStream CreateStreamWithPriority(int priority, int dev) final {
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
    return static_cast<DLStream>(stream);
   }

  void FreeStream(int dev, DLStream stream) final {
    //CUDA_CALL(cudaSetDevice(dev));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }


  void StreamSync(DLStream stream, int dev=0) final {
    //CUDA_CALL(cudaSetDevice(dev));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void CopyDataFromTo(void* from, void* to, size_t size, MemCpyDirct kind) final {
    if (kind == D2H)
      CUDA_CALL(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
    if (kind == H2D)
      CUDA_CALL(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
  }

  void CopyDataFromToAsync(void* from, void* to, size_t size, MemCpyDirct kind, DLStream stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    if (kind == D2H)
      CUDA_CALL(cudaMemcpyAsync(to, from, size, cudaMemcpyDeviceToHost, cu_stream));
    if (kind == H2D)
      CUDA_CALL(cudaMemcpyAsync(to, from, size, cudaMemcpyHostToDevice, cu_stream));
  }

  void CompileKernel(const string& model_name, const string& layer_idx_str, const string& kernel_name, const string& layer_name="") {

    string path = "/home/malixian/repos/fineco/include/kernel/cuda/" + model_name + "/";
    if (layer_idx_str != "")
      path += layer_idx_str + "/";
    path += layer_name;

    //cout<<"compile path: "<<path<<endl;
    string cd_path = "cd " + path;
    string file_name = kernel_name + ".cu";
    string cmd = cd_path + " && " "nvcc -arch=sm_80 --cubin " + file_name;
    string output = path + "/" + kernel_name + ".cubin";
    if(exists_file(output)) {
      //cout<< layer_idx_str + " " + kernel_name + " Has Exist"<<endl;
      return;
    }

    system(cmd.c_str());
    if (exists_file(output)) {
      std::cout<< layer_idx_str + " " + kernel_name + " Compile Successfully "<<std::endl;
    } else {
      std::cout<<"Kernel Compile Failed "<<std::endl;
    }
  }

  void CompileKernelByPath(const string& path, const string& kernel_name) {

    string base_path = "/home/malixian/repos/fineco/include/kernel/cuda/";

    if (path != "") {
      base_path +=  path + "/";
    }
    
    //cout<<"compile path: "<<path<<endl;
    string cd_path = "cd " + base_path;
    string file_name = kernel_name + ".cu";
    string cmd = cd_path + " && " "nvcc -arch=sm_80 --cubin " + file_name;
    string output = base_path + "/" + kernel_name + ".cubin";
    if(exists_file(output)) {
      //cout<< layer_idx_str + " " + kernel_name + " Has Exist"<<endl;
      return;
    }

    system(cmd.c_str());
    if (exists_file(output)) {
      std::cout<<  base_path + " " + kernel_name + " Compile Successfully "<<std::endl;
    } else {
      std::cout<<"Kernel Compile Failed "<<std::endl;
    }
  }



  void* GetFunction(const string& model_name, const string& layer_idx_str, const string& kernel_name, const string& layer_name="") {
    CUmodule module;
    CUfunction function;

    string path = "/home/malixian/repos/fineco/include/kernel/cuda/" + model_name + "/";
    if (layer_idx_str != "")
      path += layer_idx_str + "/";
    string module_file = "";
    string::size_type idx = layer_name.find("/");
    if ( layer_name != "" && idx == string::npos ){
      path += layer_name + "/";
    } else {
      path += layer_name;
    }
    
    
    module_file = path + kernel_name + ".cubin";
    //cout<<"get function path: "<<module_file<<endl;
    CUDA_DRIVER_CALL(cuModuleLoad(&module, module_file.c_str()));
    CUDA_DRIVER_CALL(cuModuleGetFunction(&function, module, kernel_name.c_str()));
    return function;
  }

  void* GetFunctionByPath(const string& path, const string& kernel_name) {
    CUmodule module;
    CUfunction function;

    string base_path = "/home/malixian/repos/fineco/include/kernel/cuda/";
    if (path != "")
      base_path += path + "/";
    string module_file = base_path + kernel_name + ".cubin";
    CUDA_DRIVER_CALL(cuModuleLoad(&module, module_file.c_str()));
    CUDA_DRIVER_CALL(cuModuleGetFunction(&function, module, kernel_name.c_str()));
    return function;
  }



  void LaunchKernel (void* function, size_t Grid, size_t Block, DLStream stream, void** args) {
    //cuCtxSetCurrent(_context);
    //cout<<Grid<<" "<<Block<<endl;
    CUfunction cu_function = static_cast<CUfunction>(function);
    CUstream custream = static_cast<CUstream>(stream);
    //cout<<"Grid: "<<Grid<<" Block: "<<Block<<endl;
    CUDA_DRIVER_CALL(cuLaunchKernel(cu_function, Grid, 1, 1, Block, 1, 1, 0, custream, args, 0));
    //auto ret = cuLaunchKernel(cu_function, Grid, 1, 1, Block, 1, 1, 0, custream, args, 0);
    //cout<< "launch kernel result:"<<ret<<endl;
  }

  void DeviceSync() {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  static CUDADeviceAPI* GetCUDADeviceAPI() {
    if (_inst == NULL) {
      _inst = new CUDADeviceAPI();
    } 
    return _inst;
  }

  private:
   static CUDADeviceAPI* _inst;

  private:
   CUdevice _device;
   CUcontext _context;

};

// init static member
CUDADeviceAPI* CUDADeviceAPI::_inst = NULL;

#endif
#endif