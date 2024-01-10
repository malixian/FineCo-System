#include "hip/hip_runtime.h"

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(32) candidate1(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[256];
  __shared__ float weight_shared[8192];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + ((k_outer_outer * 256) + (((int)threadIdx.x) * 2)));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(data + (((k_outer_outer * 256) + (((int)threadIdx.x) * 2)) + 64));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(data + (((k_outer_outer * 256) + (((int)threadIdx.x) * 2)) + 128));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 192)) = *(float2*)(data + (((k_outer_outer * 256) + (((int)threadIdx.x) * 2)) + 192));
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + (((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 128));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 2048));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 2048));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 4096));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 4096));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 6144));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 6144));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 8192));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1152)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 8192));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 10240));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1408)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 10240));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 12288));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1664)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 12288));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 14336));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1920)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 14336));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 16384));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2176)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 16384));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 18432));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2432)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 18432));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 20480));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2688)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 20480));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 22528));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2944)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 22528));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 24576));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 24576));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 26624));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3456)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 26624));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 28672));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3712)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 28672));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 30720));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3968)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 30720));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 32768));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4224)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 32768));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4352)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 34816));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4480)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 34816));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 36864));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4736)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 36864));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4864)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 38912));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4992)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 38912));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 40960));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5248)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 40960));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5376)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 43008));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5504)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 43008));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5632)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 45056));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5760)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 45056));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5888)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 47104));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6016)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 47104));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 49152));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6272)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 49152));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 51200));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6528)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 51200));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6656)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 53248));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6784)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 53248));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6912)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 55296));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7040)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 55296));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 57344));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7296)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 57344));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7424)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 59392));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7552)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 59392));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7680)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 61440));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7808)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 61440));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7936)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 63488));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8064)) = *(float4*)(weight + ((((((int)blockIdx.x) * 65536) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)) + 63488));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[(k_outer_inner * 64)] * weight_shared[((((int)threadIdx.x) * 256) + (k_outer_inner * 64))]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 1)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 1)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 2)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 2)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 3)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 3)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 4)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 4)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 5)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 5)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 6)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 6)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 7)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 7)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 8)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 8)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 9)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 9)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 10)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 10)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 11)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 11)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 12)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 12)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 13)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 13)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 14)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 14)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 15)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 15)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 16)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 16)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 17)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 17)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 18)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 18)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 19)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 19)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 20)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 20)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 21)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 21)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 22)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 22)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 23)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 23)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 24)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 24)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 25)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 25)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 26)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 26)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 27)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 27)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 28)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 28)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 29)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 29)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 30)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 30)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 31)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 31)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 32)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 32)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 33)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 33)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 34)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 34)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 35)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 35)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 36)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 36)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 37)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 37)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 38)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 38)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 39)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 39)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 40)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 40)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 41)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 41)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 42)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 42)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 43)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 43)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 44)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 44)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 45)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 45)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 46)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 46)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 47)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 47)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 48)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 48)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 49)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 49)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 50)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 50)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 51)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 51)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 52)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 52)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 53)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 53)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 54)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 54)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 55)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 55)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 56)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 56)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 57)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 57)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 58)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 58)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 59)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 59)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 60)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 60)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 61)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 61)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 62)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 62)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 64) + 63)] * weight_shared[(((((int)threadIdx.x) * 256) + (k_outer_inner * 64)) + 63)]));
    }
  }
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


