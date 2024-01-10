#include "hip/hip_runtime.h"

extern "C" __global__ void sleep(int us) {
    for(int i=0; i<1000; i++)
        // TODO: error: use of undeclared identifier '__nanosleep'
        __nanosleep(us); // ls
}