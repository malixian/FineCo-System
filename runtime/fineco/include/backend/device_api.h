#ifndef RUNTIME_DEVICE_API_H_
#define RUNTIME_DEVICE_API_H_

#include <stdlib.h> 
#include <string>
typedef void* DLStream;
enum MemCpyDirct {H2D, D2H};
enum DLDataType {DLFLOAT, DLINT};
using namespace std;


class  DeviceAPI {
 public:
  virtual ~DeviceAPI() {}
  
  virtual void SetDevice(int dev)=0;

  virtual void* AllocDataSpace(size_t nbytes, DLDataType type_hint)=0;
  
  virtual void FreeDataSpace(int dev, void* ptr)=0;
  
  virtual void CopyDataFromTo(void* from, void* to, size_t size, MemCpyDirct dirct)=0;
  virtual void CopyDataFromToAsync(void* from, void* to, size_t size, MemCpyDirct dirct, DLStream stream)=0;
  
  virtual DLStream CreateStream(int dev)=0;

  virtual void FreeStream(int dev, DLStream stream)=0;

  virtual void StreamSync(DLStream stream, int dev=0)=0;

  virtual void CompileKernel(const string& model_name, const string& layer_idx_str, const string& kernel_name, const string& layer_name="")=0;

  virtual void* GetFunction(const string& model_name, const string& layer_idx_str, const string& kernel_name, const string& layer_name="") = 0;

  virtual void LaunchKernel(void* function, size_t Grid, size_t Block, DLStream stream, void** args) = 0;

  virtual void CompileKernelByPath(const string& path, const string& kernel_name) = 0;

  virtual void* GetFunctionByPath(const string& path, const string& kernel_name) = 0;

  virtual DLStream CreateStreamWithPriority(int priority, int dev) = 0;

  virtual void DeviceSync() = 0;
  //virtual void SetStream(int dev, DLStream stream);

};

#endif