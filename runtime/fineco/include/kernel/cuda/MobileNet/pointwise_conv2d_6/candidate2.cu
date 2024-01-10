
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
extern "C" __global__ void __launch_bounds__(224) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[448];
  __shared__ float kernel_shared[2048];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[(((((rc_outer_outer * 3136) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 1568)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 17920)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 25088)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672)];
    if (((int)threadIdx.x) < 32) {
      kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 32256)];
    }
    __syncthreads();
    for (int ff_outer_inner = 0; ff_outer_inner < 4; ++ff_outer_inner) {
      for (int yy_outer_inner = 0; yy_outer_inner < 2; ++yy_outer_inner) {
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          conv2d_nchw[((ff_outer_inner * 2) + yy_outer_inner)] = (conv2d_nchw[((ff_outer_inner * 2) + yy_outer_inner)] + (pad_temp_shared[(((rc_inner * 28) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + (ff_outer_inner * 16)) + rc_inner)]));
          conv2d_nchw[(((ff_outer_inner * 2) + yy_outer_inner) + 8)] = (conv2d_nchw[(((ff_outer_inner * 2) + yy_outer_inner) + 8)] + (pad_temp_shared[(((rc_inner * 28) + (yy_outer_inner * 14)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 14) * 64) + (ff_outer_inner * 16)) + rc_inner) + 1024)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[(((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 14) * 784)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14))] = max(conv2d_nchw[((i1_inner * 2) + i2_inner)], 0.000000e+00f);
      compute[((((((((((int)blockIdx.x) / 7) * 25088) + ((((int)threadIdx.x) / 14) * 784)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14)) + 12544)] = max(conv2d_nchw[(((i1_inner * 2) + i2_inner) + 8)], 0.000000e+00f);
    }
  }
}


