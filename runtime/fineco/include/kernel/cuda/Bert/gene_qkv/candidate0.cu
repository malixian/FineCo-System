
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
extern "C" __global__ void __launch_bounds__(32) candidate0(float* __restrict__ I, float* __restrict__ W, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[5];
  __shared__ float I_shared[640];
  __shared__ float W_shared[4096];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  T_batch_matmul_NT_local[4] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(I_shared + (((int)threadIdx.x) * 4)) = *(float4*)(I + ((k_outer_outer * 128) + (((int)threadIdx.x) * 4)));
    *(float4*)(I_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(I + (((k_outer_outer * 128) + (((int)threadIdx.x) * 4)) + 768));
    *(float4*)(I_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(I + (((k_outer_outer * 128) + (((int)threadIdx.x) * 4)) + 1536));
    *(float4*)(I_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(I + (((k_outer_outer * 128) + (((int)threadIdx.x) * 4)) + 2304));
    *(float4*)(I_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(I + (((k_outer_outer * 128) + (((int)threadIdx.x) * 4)) + 3072));
    *(float4*)(W_shared + (((int)threadIdx.x) * 4)) = *(float4*)(W + (((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 768));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1536));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2304));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3840));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4608));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5376));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1152)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6912));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7680));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1408)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 8448));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 9216));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1664)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 9984));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 10752));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 1920)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 11520));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 12288));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2176)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 13056));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 13824));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2432)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 14592));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 15360));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2688)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 16128));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 16896));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 2944)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 17664));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 18432));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3200)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 19200));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 19968));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3456)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 20736));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 21504));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3712)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 22272));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 23040));
    *(float4*)(W_shared + ((((int)threadIdx.x) * 4) + 3968)) = *(float4*)(W + ((((((int)blockIdx.x) * 24576) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 23808));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 64; ++k_inner) {
        T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[((k_outer_inner * 64) + k_inner)] * W_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 64)) + k_inner)]));
        T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((k_outer_inner * 64) + k_inner) + 128)] * W_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 64)) + k_inner)]));
        T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (I_shared[(((k_outer_inner * 64) + k_inner) + 256)] * W_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 64)) + k_inner)]));
        T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (I_shared[(((k_outer_inner * 64) + k_inner) + 384)] * W_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 64)) + k_inner)]));
        T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (I_shared[(((k_outer_inner * 64) + k_inner) + 512)] * W_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 64)) + k_inner)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 5; ++i_inner) {
    T_batch_matmul_NT[(((i_inner * 768) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[i_inner];
  }
}


