#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

using namespace std;

CUmodule   module;
CUfunction function;
CUcontext context;
char       *module_file = (char*) "conv_relu_11x11_100.cubin";
char       *kernel_name = (char*) "conv_relu_1_100";

int main(){

    

    /*
    CUresult err = cuInit(0);

    CUdevice device;
    cuDeviceGet(&device, 0) ;

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        exit(-1);
    }
    */   

    CUresult err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        cout<<err<<endl;
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        exit(-1);
    }

}
