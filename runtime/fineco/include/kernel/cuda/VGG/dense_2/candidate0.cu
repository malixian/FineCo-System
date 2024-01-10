

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
extern "C" __global__ void __launch_bounds__(128) candidate0(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[64];
  __shared__ float weight_shared[8192];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      data_shared[((int)threadIdx.x)] = data[((k_outer_outer * 64) + ((int)threadIdx.x))];
    }
    *(float4*)(weight_shared + (((int)threadIdx.x) * 4)) = *(float4*)(weight + ((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 32768));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 65536));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 98304));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 131072));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 163840));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 196608));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 229376));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 262144));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 294912));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 327680));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 5632)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 360448));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 393216));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 6656)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 425984));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 458752));
    *(float4*)(weight_shared + ((((int)threadIdx.x) * 4) + 7680)) = *(float4*)(weight + (((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 4) * 4096)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 491520));
    __syncthreads();
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[0] * weight_shared[(((int)threadIdx.x) * 64)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[1] * weight_shared[((((int)threadIdx.x) * 64) + 1)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[2] * weight_shared[((((int)threadIdx.x) * 64) + 2)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[3] * weight_shared[((((int)threadIdx.x) * 64) + 3)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[4] * weight_shared[((((int)threadIdx.x) * 64) + 4)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[5] * weight_shared[((((int)threadIdx.x) * 64) + 5)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[6] * weight_shared[((((int)threadIdx.x) * 64) + 6)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[7] * weight_shared[((((int)threadIdx.x) * 64) + 7)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[8] * weight_shared[((((int)threadIdx.x) * 64) + 8)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[9] * weight_shared[((((int)threadIdx.x) * 64) + 9)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[10] * weight_shared[((((int)threadIdx.x) * 64) + 10)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[11] * weight_shared[((((int)threadIdx.x) * 64) + 11)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[12] * weight_shared[((((int)threadIdx.x) * 64) + 12)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[13] * weight_shared[((((int)threadIdx.x) * 64) + 13)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[14] * weight_shared[((((int)threadIdx.x) * 64) + 14)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[15] * weight_shared[((((int)threadIdx.x) * 64) + 15)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[16] * weight_shared[((((int)threadIdx.x) * 64) + 16)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[17] * weight_shared[((((int)threadIdx.x) * 64) + 17)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[18] * weight_shared[((((int)threadIdx.x) * 64) + 18)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[19] * weight_shared[((((int)threadIdx.x) * 64) + 19)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[20] * weight_shared[((((int)threadIdx.x) * 64) + 20)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[21] * weight_shared[((((int)threadIdx.x) * 64) + 21)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[22] * weight_shared[((((int)threadIdx.x) * 64) + 22)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[23] * weight_shared[((((int)threadIdx.x) * 64) + 23)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[24] * weight_shared[((((int)threadIdx.x) * 64) + 24)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[25] * weight_shared[((((int)threadIdx.x) * 64) + 25)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[26] * weight_shared[((((int)threadIdx.x) * 64) + 26)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[27] * weight_shared[((((int)threadIdx.x) * 64) + 27)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[28] * weight_shared[((((int)threadIdx.x) * 64) + 28)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[29] * weight_shared[((((int)threadIdx.x) * 64) + 29)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[30] * weight_shared[((((int)threadIdx.x) * 64) + 30)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[31] * weight_shared[((((int)threadIdx.x) * 64) + 31)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[32] * weight_shared[((((int)threadIdx.x) * 64) + 32)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[33] * weight_shared[((((int)threadIdx.x) * 64) + 33)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[34] * weight_shared[((((int)threadIdx.x) * 64) + 34)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[35] * weight_shared[((((int)threadIdx.x) * 64) + 35)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[36] * weight_shared[((((int)threadIdx.x) * 64) + 36)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[37] * weight_shared[((((int)threadIdx.x) * 64) + 37)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[38] * weight_shared[((((int)threadIdx.x) * 64) + 38)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[39] * weight_shared[((((int)threadIdx.x) * 64) + 39)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[40] * weight_shared[((((int)threadIdx.x) * 64) + 40)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[41] * weight_shared[((((int)threadIdx.x) * 64) + 41)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[42] * weight_shared[((((int)threadIdx.x) * 64) + 42)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[43] * weight_shared[((((int)threadIdx.x) * 64) + 43)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[44] * weight_shared[((((int)threadIdx.x) * 64) + 44)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[45] * weight_shared[((((int)threadIdx.x) * 64) + 45)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[46] * weight_shared[((((int)threadIdx.x) * 64) + 46)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[47] * weight_shared[((((int)threadIdx.x) * 64) + 47)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[48] * weight_shared[((((int)threadIdx.x) * 64) + 48)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[49] * weight_shared[((((int)threadIdx.x) * 64) + 49)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[50] * weight_shared[((((int)threadIdx.x) * 64) + 50)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[51] * weight_shared[((((int)threadIdx.x) * 64) + 51)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[52] * weight_shared[((((int)threadIdx.x) * 64) + 52)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[53] * weight_shared[((((int)threadIdx.x) * 64) + 53)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[54] * weight_shared[((((int)threadIdx.x) * 64) + 54)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[55] * weight_shared[((((int)threadIdx.x) * 64) + 55)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[56] * weight_shared[((((int)threadIdx.x) * 64) + 56)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[57] * weight_shared[((((int)threadIdx.x) * 64) + 57)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[58] * weight_shared[((((int)threadIdx.x) * 64) + 58)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[59] * weight_shared[((((int)threadIdx.x) * 64) + 59)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[60] * weight_shared[((((int)threadIdx.x) * 64) + 60)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[61] * weight_shared[((((int)threadIdx.x) * 64) + 61)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[62] * weight_shared[((((int)threadIdx.x) * 64) + 62)]));
    T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[63] * weight_shared[((((int)threadIdx.x) * 64) + 63)]));
  }
  compute[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
}


