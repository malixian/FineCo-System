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
extern "C" __global__ void __launch_bounds__(50) candidate0(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[196];
  __shared__ float weight_shared[9800];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 49) {
      *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + ((k_outer_outer * 196) + (((int)threadIdx.x) * 4)));
    }
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((int)threadIdx.x) / 49) * 25088)) + (k_outer_outer * 196)) + ((((int)threadIdx.x) % 49) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 4) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 8) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 12) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 16) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 1000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 20) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 1200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 24) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 1400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 28) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 1600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 32) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 1800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 36) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 2000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 40) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 2200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 44) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 2400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 48) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 2600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 52) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 2800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 56) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 3000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 60) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 3200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 64) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 3400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 68) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 3600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 72) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 3800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 76) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 4000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 80) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 4200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 84) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 4400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 88) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 4600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 92) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 4800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 96) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 5000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 100) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 5200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 104) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 5400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 108) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 5600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 112) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 5800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 116) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 6000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 120) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 6200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 124) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 6400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 128) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 6600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 132) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 6800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 136) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 7000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 140) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 7200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 144) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 7400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 148) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 7600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 152) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 7800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 156) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 8000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 160) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 8200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 164) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 8400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 168) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 8600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 172) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 8800)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 8800) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 176) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 9000)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 9000) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 180) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 9200)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 9200) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 184) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 9400)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 9400) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 188) % 196)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 9600)) = *(float4*)(weight + ((((((int)blockIdx.x) * 1254400) + ((((((int)threadIdx.x) * 4) + 9600) / 196) * 25088)) + (k_outer_outer * 196)) + (((((int)threadIdx.x) * 4) + 192) % 196)));
    __syncthreads();
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[0] * weight_shared[(((int)threadIdx.x) * 196)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[1] * weight_shared[((((int)threadIdx.x) * 196) + 1)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[2] * weight_shared[((((int)threadIdx.x) * 196) + 2)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[3] * weight_shared[((((int)threadIdx.x) * 196) + 3)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[4] * weight_shared[((((int)threadIdx.x) * 196) + 4)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[5] * weight_shared[((((int)threadIdx.x) * 196) + 5)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[6] * weight_shared[((((int)threadIdx.x) * 196) + 6)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[7] * weight_shared[((((int)threadIdx.x) * 196) + 7)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[8] * weight_shared[((((int)threadIdx.x) * 196) + 8)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[9] * weight_shared[((((int)threadIdx.x) * 196) + 9)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[10] * weight_shared[((((int)threadIdx.x) * 196) + 10)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[11] * weight_shared[((((int)threadIdx.x) * 196) + 11)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[12] * weight_shared[((((int)threadIdx.x) * 196) + 12)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[13] * weight_shared[((((int)threadIdx.x) * 196) + 13)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[14] * weight_shared[((((int)threadIdx.x) * 196) + 14)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[15] * weight_shared[((((int)threadIdx.x) * 196) + 15)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[16] * weight_shared[((((int)threadIdx.x) * 196) + 16)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[17] * weight_shared[((((int)threadIdx.x) * 196) + 17)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[18] * weight_shared[((((int)threadIdx.x) * 196) + 18)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[19] * weight_shared[((((int)threadIdx.x) * 196) + 19)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[20] * weight_shared[((((int)threadIdx.x) * 196) + 20)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[21] * weight_shared[((((int)threadIdx.x) * 196) + 21)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[22] * weight_shared[((((int)threadIdx.x) * 196) + 22)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[23] * weight_shared[((((int)threadIdx.x) * 196) + 23)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[24] * weight_shared[((((int)threadIdx.x) * 196) + 24)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[25] * weight_shared[((((int)threadIdx.x) * 196) + 25)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[26] * weight_shared[((((int)threadIdx.x) * 196) + 26)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[27] * weight_shared[((((int)threadIdx.x) * 196) + 27)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[28] * weight_shared[((((int)threadIdx.x) * 196) + 28)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[29] * weight_shared[((((int)threadIdx.x) * 196) + 29)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[30] * weight_shared[((((int)threadIdx.x) * 196) + 30)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[31] * weight_shared[((((int)threadIdx.x) * 196) + 31)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[32] * weight_shared[((((int)threadIdx.x) * 196) + 32)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[33] * weight_shared[((((int)threadIdx.x) * 196) + 33)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[34] * weight_shared[((((int)threadIdx.x) * 196) + 34)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[35] * weight_shared[((((int)threadIdx.x) * 196) + 35)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[36] * weight_shared[((((int)threadIdx.x) * 196) + 36)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[37] * weight_shared[((((int)threadIdx.x) * 196) + 37)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[38] * weight_shared[((((int)threadIdx.x) * 196) + 38)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[39] * weight_shared[((((int)threadIdx.x) * 196) + 39)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[40] * weight_shared[((((int)threadIdx.x) * 196) + 40)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[41] * weight_shared[((((int)threadIdx.x) * 196) + 41)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[42] * weight_shared[((((int)threadIdx.x) * 196) + 42)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[43] * weight_shared[((((int)threadIdx.x) * 196) + 43)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[44] * weight_shared[((((int)threadIdx.x) * 196) + 44)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[45] * weight_shared[((((int)threadIdx.x) * 196) + 45)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[46] * weight_shared[((((int)threadIdx.x) * 196) + 46)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[47] * weight_shared[((((int)threadIdx.x) * 196) + 47)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[48] * weight_shared[((((int)threadIdx.x) * 196) + 48)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[49] * weight_shared[((((int)threadIdx.x) * 196) + 49)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[50] * weight_shared[((((int)threadIdx.x) * 196) + 50)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[51] * weight_shared[((((int)threadIdx.x) * 196) + 51)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[52] * weight_shared[((((int)threadIdx.x) * 196) + 52)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[53] * weight_shared[((((int)threadIdx.x) * 196) + 53)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[54] * weight_shared[((((int)threadIdx.x) * 196) + 54)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[55] * weight_shared[((((int)threadIdx.x) * 196) + 55)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[56] * weight_shared[((((int)threadIdx.x) * 196) + 56)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[57] * weight_shared[((((int)threadIdx.x) * 196) + 57)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[58] * weight_shared[((((int)threadIdx.x) * 196) + 58)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[59] * weight_shared[((((int)threadIdx.x) * 196) + 59)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[60] * weight_shared[((((int)threadIdx.x) * 196) + 60)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[61] * weight_shared[((((int)threadIdx.x) * 196) + 61)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[62] * weight_shared[((((int)threadIdx.x) * 196) + 62)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[63] * weight_shared[((((int)threadIdx.x) * 196) + 63)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[64] * weight_shared[((((int)threadIdx.x) * 196) + 64)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[65] * weight_shared[((((int)threadIdx.x) * 196) + 65)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[66] * weight_shared[((((int)threadIdx.x) * 196) + 66)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[67] * weight_shared[((((int)threadIdx.x) * 196) + 67)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[68] * weight_shared[((((int)threadIdx.x) * 196) + 68)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[69] * weight_shared[((((int)threadIdx.x) * 196) + 69)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[70] * weight_shared[((((int)threadIdx.x) * 196) + 70)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[71] * weight_shared[((((int)threadIdx.x) * 196) + 71)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[72] * weight_shared[((((int)threadIdx.x) * 196) + 72)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[73] * weight_shared[((((int)threadIdx.x) * 196) + 73)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[74] * weight_shared[((((int)threadIdx.x) * 196) + 74)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[75] * weight_shared[((((int)threadIdx.x) * 196) + 75)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[76] * weight_shared[((((int)threadIdx.x) * 196) + 76)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[77] * weight_shared[((((int)threadIdx.x) * 196) + 77)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[78] * weight_shared[((((int)threadIdx.x) * 196) + 78)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[79] * weight_shared[((((int)threadIdx.x) * 196) + 79)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[80] * weight_shared[((((int)threadIdx.x) * 196) + 80)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[81] * weight_shared[((((int)threadIdx.x) * 196) + 81)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[82] * weight_shared[((((int)threadIdx.x) * 196) + 82)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[83] * weight_shared[((((int)threadIdx.x) * 196) + 83)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[84] * weight_shared[((((int)threadIdx.x) * 196) + 84)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[85] * weight_shared[((((int)threadIdx.x) * 196) + 85)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[86] * weight_shared[((((int)threadIdx.x) * 196) + 86)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[87] * weight_shared[((((int)threadIdx.x) * 196) + 87)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[88] * weight_shared[((((int)threadIdx.x) * 196) + 88)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[89] * weight_shared[((((int)threadIdx.x) * 196) + 89)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[90] * weight_shared[((((int)threadIdx.x) * 196) + 90)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[91] * weight_shared[((((int)threadIdx.x) * 196) + 91)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[92] * weight_shared[((((int)threadIdx.x) * 196) + 92)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[93] * weight_shared[((((int)threadIdx.x) * 196) + 93)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[94] * weight_shared[((((int)threadIdx.x) * 196) + 94)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[95] * weight_shared[((((int)threadIdx.x) * 196) + 95)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[96] * weight_shared[((((int)threadIdx.x) * 196) + 96)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[97] * weight_shared[((((int)threadIdx.x) * 196) + 97)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[98] * weight_shared[((((int)threadIdx.x) * 196) + 98)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[99] * weight_shared[((((int)threadIdx.x) * 196) + 99)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[100] * weight_shared[((((int)threadIdx.x) * 196) + 100)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[101] * weight_shared[((((int)threadIdx.x) * 196) + 101)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[102] * weight_shared[((((int)threadIdx.x) * 196) + 102)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[103] * weight_shared[((((int)threadIdx.x) * 196) + 103)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[104] * weight_shared[((((int)threadIdx.x) * 196) + 104)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[105] * weight_shared[((((int)threadIdx.x) * 196) + 105)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[106] * weight_shared[((((int)threadIdx.x) * 196) + 106)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[107] * weight_shared[((((int)threadIdx.x) * 196) + 107)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[108] * weight_shared[((((int)threadIdx.x) * 196) + 108)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[109] * weight_shared[((((int)threadIdx.x) * 196) + 109)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[110] * weight_shared[((((int)threadIdx.x) * 196) + 110)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[111] * weight_shared[((((int)threadIdx.x) * 196) + 111)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[112] * weight_shared[((((int)threadIdx.x) * 196) + 112)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[113] * weight_shared[((((int)threadIdx.x) * 196) + 113)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[114] * weight_shared[((((int)threadIdx.x) * 196) + 114)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[115] * weight_shared[((((int)threadIdx.x) * 196) + 115)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[116] * weight_shared[((((int)threadIdx.x) * 196) + 116)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[117] * weight_shared[((((int)threadIdx.x) * 196) + 117)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[118] * weight_shared[((((int)threadIdx.x) * 196) + 118)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[119] * weight_shared[((((int)threadIdx.x) * 196) + 119)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[120] * weight_shared[((((int)threadIdx.x) * 196) + 120)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[121] * weight_shared[((((int)threadIdx.x) * 196) + 121)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[122] * weight_shared[((((int)threadIdx.x) * 196) + 122)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[123] * weight_shared[((((int)threadIdx.x) * 196) + 123)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[124] * weight_shared[((((int)threadIdx.x) * 196) + 124)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[125] * weight_shared[((((int)threadIdx.x) * 196) + 125)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[126] * weight_shared[((((int)threadIdx.x) * 196) + 126)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[127] * weight_shared[((((int)threadIdx.x) * 196) + 127)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[128] * weight_shared[((((int)threadIdx.x) * 196) + 128)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[129] * weight_shared[((((int)threadIdx.x) * 196) + 129)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[130] * weight_shared[((((int)threadIdx.x) * 196) + 130)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[131] * weight_shared[((((int)threadIdx.x) * 196) + 131)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[132] * weight_shared[((((int)threadIdx.x) * 196) + 132)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[133] * weight_shared[((((int)threadIdx.x) * 196) + 133)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[134] * weight_shared[((((int)threadIdx.x) * 196) + 134)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[135] * weight_shared[((((int)threadIdx.x) * 196) + 135)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[136] * weight_shared[((((int)threadIdx.x) * 196) + 136)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[137] * weight_shared[((((int)threadIdx.x) * 196) + 137)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[138] * weight_shared[((((int)threadIdx.x) * 196) + 138)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[139] * weight_shared[((((int)threadIdx.x) * 196) + 139)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[140] * weight_shared[((((int)threadIdx.x) * 196) + 140)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[141] * weight_shared[((((int)threadIdx.x) * 196) + 141)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[142] * weight_shared[((((int)threadIdx.x) * 196) + 142)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[143] * weight_shared[((((int)threadIdx.x) * 196) + 143)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[144] * weight_shared[((((int)threadIdx.x) * 196) + 144)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[145] * weight_shared[((((int)threadIdx.x) * 196) + 145)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[146] * weight_shared[((((int)threadIdx.x) * 196) + 146)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[147] * weight_shared[((((int)threadIdx.x) * 196) + 147)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[148] * weight_shared[((((int)threadIdx.x) * 196) + 148)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[149] * weight_shared[((((int)threadIdx.x) * 196) + 149)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[150] * weight_shared[((((int)threadIdx.x) * 196) + 150)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[151] * weight_shared[((((int)threadIdx.x) * 196) + 151)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[152] * weight_shared[((((int)threadIdx.x) * 196) + 152)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[153] * weight_shared[((((int)threadIdx.x) * 196) + 153)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[154] * weight_shared[((((int)threadIdx.x) * 196) + 154)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[155] * weight_shared[((((int)threadIdx.x) * 196) + 155)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[156] * weight_shared[((((int)threadIdx.x) * 196) + 156)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[157] * weight_shared[((((int)threadIdx.x) * 196) + 157)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[158] * weight_shared[((((int)threadIdx.x) * 196) + 158)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[159] * weight_shared[((((int)threadIdx.x) * 196) + 159)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[160] * weight_shared[((((int)threadIdx.x) * 196) + 160)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[161] * weight_shared[((((int)threadIdx.x) * 196) + 161)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[162] * weight_shared[((((int)threadIdx.x) * 196) + 162)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[163] * weight_shared[((((int)threadIdx.x) * 196) + 163)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[164] * weight_shared[((((int)threadIdx.x) * 196) + 164)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[165] * weight_shared[((((int)threadIdx.x) * 196) + 165)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[166] * weight_shared[((((int)threadIdx.x) * 196) + 166)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[167] * weight_shared[((((int)threadIdx.x) * 196) + 167)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[168] * weight_shared[((((int)threadIdx.x) * 196) + 168)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[169] * weight_shared[((((int)threadIdx.x) * 196) + 169)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[170] * weight_shared[((((int)threadIdx.x) * 196) + 170)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[171] * weight_shared[((((int)threadIdx.x) * 196) + 171)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[172] * weight_shared[((((int)threadIdx.x) * 196) + 172)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[173] * weight_shared[((((int)threadIdx.x) * 196) + 173)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[174] * weight_shared[((((int)threadIdx.x) * 196) + 174)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[175] * weight_shared[((((int)threadIdx.x) * 196) + 175)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[176] * weight_shared[((((int)threadIdx.x) * 196) + 176)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[177] * weight_shared[((((int)threadIdx.x) * 196) + 177)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[178] * weight_shared[((((int)threadIdx.x) * 196) + 178)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[179] * weight_shared[((((int)threadIdx.x) * 196) + 179)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[180] * weight_shared[((((int)threadIdx.x) * 196) + 180)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[181] * weight_shared[((((int)threadIdx.x) * 196) + 181)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[182] * weight_shared[((((int)threadIdx.x) * 196) + 182)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[183] * weight_shared[((((int)threadIdx.x) * 196) + 183)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[184] * weight_shared[((((int)threadIdx.x) * 196) + 184)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[185] * weight_shared[((((int)threadIdx.x) * 196) + 185)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[186] * weight_shared[((((int)threadIdx.x) * 196) + 186)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[187] * weight_shared[((((int)threadIdx.x) * 196) + 187)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[188] * weight_shared[((((int)threadIdx.x) * 196) + 188)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[189] * weight_shared[((((int)threadIdx.x) * 196) + 189)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[190] * weight_shared[((((int)threadIdx.x) * 196) + 190)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[191] * weight_shared[((((int)threadIdx.x) * 196) + 191)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[192] * weight_shared[((((int)threadIdx.x) * 196) + 192)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[193] * weight_shared[((((int)threadIdx.x) * 196) + 193)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[194] * weight_shared[((((int)threadIdx.x) * 196) + 194)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[195] * weight_shared[((((int)threadIdx.x) * 196) + 195)]));
  }
  compute[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))]);
}


