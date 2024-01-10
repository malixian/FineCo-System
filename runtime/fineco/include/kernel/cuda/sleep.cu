extern "C" __global__ void sleep(int us) {
    for(int i=0; i<1000; i++)
        __nanosleep(us); // ls
}