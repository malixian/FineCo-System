
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
extern "C" __global__ void __launch_bounds__(96) candidate1(float* __restrict__ I, float* __restrict__ W, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[5];
  __shared__ float I_shared[40];
  __shared__ float W_shared[768];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  T_batch_matmul_NT_local[4] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 96; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 40) {
      I_shared[((int)threadIdx.x)] = I[((((((int)threadIdx.x) >> 3) * 768) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    *(float2*)(W_shared + (((int)threadIdx.x) * 2)) = *(float2*)(W + ((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) >> 2) * 768)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    *(float2*)(W_shared + ((((int)threadIdx.x) * 2) + 192)) = *(float2*)(W + (((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) >> 2) * 768)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 18432));
    *(float2*)(W_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(W + (((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) >> 2) * 768)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 36864));
    *(float2*)(W_shared + ((((int)threadIdx.x) * 2) + 576)) = *(float2*)(W + (((((((int)blockIdx.x) * 73728) + ((((int)threadIdx.x) >> 2) * 768)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 55296));
    __syncthreads();
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[0] * W_shared[(((int)threadIdx.x) * 8)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[8] * W_shared[(((int)threadIdx.x) * 8)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[16] * W_shared[(((int)threadIdx.x) * 8)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[24] * W_shared[(((int)threadIdx.x) * 8)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[32] * W_shared[(((int)threadIdx.x) * 8)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[1] * W_shared[((((int)threadIdx.x) * 8) + 1)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[9] * W_shared[((((int)threadIdx.x) * 8) + 1)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[17] * W_shared[((((int)threadIdx.x) * 8) + 1)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[25] * W_shared[((((int)threadIdx.x) * 8) + 1)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[33] * W_shared[((((int)threadIdx.x) * 8) + 1)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[2] * W_shared[((((int)threadIdx.x) * 8) + 2)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[10] * W_shared[((((int)threadIdx.x) * 8) + 2)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[18] * W_shared[((((int)threadIdx.x) * 8) + 2)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[26] * W_shared[((((int)threadIdx.x) * 8) + 2)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[34] * W_shared[((((int)threadIdx.x) * 8) + 2)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[3] * W_shared[((((int)threadIdx.x) * 8) + 3)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[11] * W_shared[((((int)threadIdx.x) * 8) + 3)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[19] * W_shared[((((int)threadIdx.x) * 8) + 3)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[27] * W_shared[((((int)threadIdx.x) * 8) + 3)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[35] * W_shared[((((int)threadIdx.x) * 8) + 3)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[4] * W_shared[((((int)threadIdx.x) * 8) + 4)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[12] * W_shared[((((int)threadIdx.x) * 8) + 4)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[20] * W_shared[((((int)threadIdx.x) * 8) + 4)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[28] * W_shared[((((int)threadIdx.x) * 8) + 4)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[36] * W_shared[((((int)threadIdx.x) * 8) + 4)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[5] * W_shared[((((int)threadIdx.x) * 8) + 5)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[13] * W_shared[((((int)threadIdx.x) * 8) + 5)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[21] * W_shared[((((int)threadIdx.x) * 8) + 5)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[29] * W_shared[((((int)threadIdx.x) * 8) + 5)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[37] * W_shared[((((int)threadIdx.x) * 8) + 5)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[6] * W_shared[((((int)threadIdx.x) * 8) + 6)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[14] * W_shared[((((int)threadIdx.x) * 8) + 6)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[22] * W_shared[((((int)threadIdx.x) * 8) + 6)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[30] * W_shared[((((int)threadIdx.x) * 8) + 6)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[38] * W_shared[((((int)threadIdx.x) * 8) + 6)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[7] * W_shared[((((int)threadIdx.x) * 8) + 7)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[15] * W_shared[((((int)threadIdx.x) * 8) + 7)]));
    T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[23] * W_shared[((((int)threadIdx.x) * 8) + 7)]));
    T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[31] * W_shared[((((int)threadIdx.x) * 8) + 7)]));
    T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[39] * W_shared[((((int)threadIdx.x) * 8) + 7)]));
  }
  for (int i_inner = 0; i_inner < 5; ++i_inner) {
    T_batch_matmul_NT[(((i_inner * 3072) + (((int)blockIdx.x) * 96)) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[i_inner];
  }
}


