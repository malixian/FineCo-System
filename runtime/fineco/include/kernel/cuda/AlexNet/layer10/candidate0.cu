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
  float T_matmul_NT[1];
  __shared__ float data_shared[128];
  __shared__ float weight_shared[8192];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + ((k_outer_outer * 128) + (((int)threadIdx.x) * 2)));
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(weight_shared + ((ax0_ax1_fused_outer_outer * 256) + (((int)threadIdx.x) * 4))) = *(float4*)(weight + (((((((int)blockIdx.x) * 131072) + (ax0_ax1_fused_outer_outer * 4096)) + ((((int)threadIdx.x) >> 5) * 2048)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[(k_outer_inner * 8)] * weight_shared[((((int)threadIdx.x) * 128) + (k_outer_inner * 8))]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 1)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 1)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 2)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 2)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 3)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 3)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 4)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 4)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 5)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 5)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 6)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 6)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 8) + 7)] * weight_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + 7)]));
    }
  }
  compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}


