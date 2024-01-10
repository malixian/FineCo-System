
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
extern "C" __global__ void __launch_bounds__(120) candidate0(float* __restrict__ I, float* __restrict__ W, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[2];
  __shared__ float I_shared[120];
  __shared__ float W_shared[1152];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 30) {
      *(float4*)(I_shared + (((int)threadIdx.x) * 4)) = *(float4*)(I + ((((((int)threadIdx.x) / 6) * 768) + (k_outer_outer * 24)) + ((((int)threadIdx.x) % 6) * 4)));
    }
    W_shared[((int)threadIdx.x)] = W[((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24))];
    W_shared[(((int)threadIdx.x) + 120)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 3840)];
    W_shared[(((int)threadIdx.x) + 240)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 7680)];
    W_shared[(((int)threadIdx.x) + 360)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 11520)];
    W_shared[(((int)threadIdx.x) + 480)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 15360)];
    W_shared[(((int)threadIdx.x) + 600)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 19200)];
    W_shared[(((int)threadIdx.x) + 720)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 23040)];
    W_shared[(((int)threadIdx.x) + 840)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 26880)];
    W_shared[(((int)threadIdx.x) + 960)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 30720)];
    if (((int)threadIdx.x) < 72) {
      W_shared[(((int)threadIdx.x) + 1080)] = W[(((((((int)blockIdx.x) * 36864) + ((((int)threadIdx.x) / 24) * 768)) + (k_outer_outer * 24)) + (((int)threadIdx.x) % 24)) + 34560)];
    }
    __syncthreads();
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[((((int)threadIdx.x) / 24) * 24)] * W_shared[((((int)threadIdx.x) % 24) * 24)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[((((int)threadIdx.x) / 24) * 24)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 576)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 1)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 1)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 1)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 577)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 2)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 2)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 2)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 578)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 3)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 3)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 3)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 579)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 4)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 4)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 4)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 580)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 5)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 5)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 5)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 581)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 6)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 6)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 6)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 582)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 7)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 7)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 7)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 583)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 8)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 8)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 8)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 584)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 9)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 9)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 9)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 585)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 10)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 10)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 10)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 586)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 11)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 11)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 11)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 587)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 12)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 12)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 12)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 588)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 13)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 13)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 13)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 589)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 14)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 14)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 14)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 590)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 15)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 15)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 15)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 591)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 16)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 16)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 16)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 592)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 17)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 17)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 17)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 593)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 18)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 18)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 18)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 594)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 19)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 19)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 19)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 595)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 20)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 20)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 20)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 596)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 21)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 21)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 21)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 597)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 22)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 22)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 22)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 598)]));
    T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 23)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 23)]));
    T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (I_shared[(((((int)threadIdx.x) / 24) * 24) + 23)] * W_shared[(((((int)threadIdx.x) % 24) * 24) + 599)]));
  }
  T_batch_matmul_NT[((((((int)threadIdx.x) / 24) * 3072) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) % 24))] = T_batch_matmul_NT_local[0];
  T_batch_matmul_NT[(((((((int)threadIdx.x) / 24) * 3072) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.x) % 24)) + 24)] = T_batch_matmul_NT_local[1];
}


