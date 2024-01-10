
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
extern "C" __global__ void __launch_bounds__(64) candidate0(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[2];
  __shared__ float data_shared[32];
  __shared__ float weight_shared[4096];
  T_matmul_NT[0] = 0.000000e+00f;
  T_matmul_NT[1] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 784; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      data_shared[((int)threadIdx.x)] = data[((k_outer_outer * 32) + ((int)threadIdx.x))];
    }
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 200704));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 401408));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 602112));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 802816));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1003520));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1204224));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1404928));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1605632));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1806336));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2007040));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2207744));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2408448));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2609152));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2809856));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(weight + (((((((int)blockIdx.x) * 3211264) + ((((int)threadIdx.x) >> 3) * 25088)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 3010560));
    __syncthreads();
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[0] * weight_shared[(((int)threadIdx.x) * 32)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[0] * weight_shared[((((int)threadIdx.x) * 32) + 2048)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[1] * weight_shared[((((int)threadIdx.x) * 32) + 1)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[1] * weight_shared[((((int)threadIdx.x) * 32) + 2049)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[2] * weight_shared[((((int)threadIdx.x) * 32) + 2)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[2] * weight_shared[((((int)threadIdx.x) * 32) + 2050)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[3] * weight_shared[((((int)threadIdx.x) * 32) + 3)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[3] * weight_shared[((((int)threadIdx.x) * 32) + 2051)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[4] * weight_shared[((((int)threadIdx.x) * 32) + 4)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[4] * weight_shared[((((int)threadIdx.x) * 32) + 2052)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[5] * weight_shared[((((int)threadIdx.x) * 32) + 5)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[5] * weight_shared[((((int)threadIdx.x) * 32) + 2053)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[6] * weight_shared[((((int)threadIdx.x) * 32) + 6)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[6] * weight_shared[((((int)threadIdx.x) * 32) + 2054)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[7] * weight_shared[((((int)threadIdx.x) * 32) + 7)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[7] * weight_shared[((((int)threadIdx.x) * 32) + 2055)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[8] * weight_shared[((((int)threadIdx.x) * 32) + 8)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[8] * weight_shared[((((int)threadIdx.x) * 32) + 2056)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[9] * weight_shared[((((int)threadIdx.x) * 32) + 9)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[9] * weight_shared[((((int)threadIdx.x) * 32) + 2057)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[10] * weight_shared[((((int)threadIdx.x) * 32) + 10)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[10] * weight_shared[((((int)threadIdx.x) * 32) + 2058)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[11] * weight_shared[((((int)threadIdx.x) * 32) + 11)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[11] * weight_shared[((((int)threadIdx.x) * 32) + 2059)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[12] * weight_shared[((((int)threadIdx.x) * 32) + 12)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[12] * weight_shared[((((int)threadIdx.x) * 32) + 2060)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[13] * weight_shared[((((int)threadIdx.x) * 32) + 13)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[13] * weight_shared[((((int)threadIdx.x) * 32) + 2061)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[14] * weight_shared[((((int)threadIdx.x) * 32) + 14)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[14] * weight_shared[((((int)threadIdx.x) * 32) + 2062)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[15] * weight_shared[((((int)threadIdx.x) * 32) + 15)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[15] * weight_shared[((((int)threadIdx.x) * 32) + 2063)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[16] * weight_shared[((((int)threadIdx.x) * 32) + 16)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[16] * weight_shared[((((int)threadIdx.x) * 32) + 2064)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[17] * weight_shared[((((int)threadIdx.x) * 32) + 17)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[17] * weight_shared[((((int)threadIdx.x) * 32) + 2065)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[18] * weight_shared[((((int)threadIdx.x) * 32) + 18)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[18] * weight_shared[((((int)threadIdx.x) * 32) + 2066)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[19] * weight_shared[((((int)threadIdx.x) * 32) + 19)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[19] * weight_shared[((((int)threadIdx.x) * 32) + 2067)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[20] * weight_shared[((((int)threadIdx.x) * 32) + 20)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[20] * weight_shared[((((int)threadIdx.x) * 32) + 2068)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[21] * weight_shared[((((int)threadIdx.x) * 32) + 21)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[21] * weight_shared[((((int)threadIdx.x) * 32) + 2069)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[22] * weight_shared[((((int)threadIdx.x) * 32) + 22)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[22] * weight_shared[((((int)threadIdx.x) * 32) + 2070)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[23] * weight_shared[((((int)threadIdx.x) * 32) + 23)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[23] * weight_shared[((((int)threadIdx.x) * 32) + 2071)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[24] * weight_shared[((((int)threadIdx.x) * 32) + 24)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[24] * weight_shared[((((int)threadIdx.x) * 32) + 2072)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[25] * weight_shared[((((int)threadIdx.x) * 32) + 25)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[25] * weight_shared[((((int)threadIdx.x) * 32) + 2073)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[26] * weight_shared[((((int)threadIdx.x) * 32) + 26)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[26] * weight_shared[((((int)threadIdx.x) * 32) + 2074)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[27] * weight_shared[((((int)threadIdx.x) * 32) + 27)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[27] * weight_shared[((((int)threadIdx.x) * 32) + 2075)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[28] * weight_shared[((((int)threadIdx.x) * 32) + 28)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[28] * weight_shared[((((int)threadIdx.x) * 32) + 2076)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[29] * weight_shared[((((int)threadIdx.x) * 32) + 29)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[29] * weight_shared[((((int)threadIdx.x) * 32) + 2077)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[30] * weight_shared[((((int)threadIdx.x) * 32) + 30)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[30] * weight_shared[((((int)threadIdx.x) * 32) + 2078)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[31] * weight_shared[((((int)threadIdx.x) * 32) + 31)]));
    T_matmul_NT[1] = (T_matmul_NT[1] + (data_shared[31] * weight_shared[((((int)threadIdx.x) * 32) + 2079)]));
  }
  compute[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  compute[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 64)] = (T_matmul_NT[1] + bias[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 64)]);
}


