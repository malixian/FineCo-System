
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
extern "C" __global__ void __launch_bounds__(448) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[64];
  __shared__ float pad_temp_shared[1792];
  __shared__ float kernel_shared[4096];
  for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 16; ++ff_c_outer_inner_init) {
    conv2d_nchw_local[(ff_c_outer_inner_init * 4)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 4) + 2)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 4) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((ff_c_outer_inner_init * 4) + 3)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 112) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) % 112))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 112) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) % 112)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[(((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 112) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) % 112)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[(((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 112) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) % 112)) + 37632)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 5376)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 8960)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[(((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 16128)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 16; ++ff_c_outer_inner) {
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[((rc_outer_inner * 448) + (((int)threadIdx.x) % 56))] * kernel_shared[((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4))]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[((rc_outer_inner * 448) + (((int)threadIdx.x) % 56))] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 16)]));
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 112)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 112)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 17)]));
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 224)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 224)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 18)]));
        conv2d_nchw_local[(ff_c_outer_inner * 4)] = (conv2d_nchw_local[(ff_c_outer_inner * 4)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 336)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 336)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 19)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4))]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 56)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 16)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 168)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 168)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 17)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 280)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 280)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 18)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 392)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((rc_outer_inner * 448) + (((int)threadIdx.x) % 56)) + 392)] * kernel_shared[(((((((int)threadIdx.x) / 56) * 512) + (ff_c_outer_inner * 32)) + (rc_outer_inner * 4)) + 19)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 32; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[((((((((int)threadIdx.x) / 56) * 100352) + (ff_inner * 3136)) + (((int)blockIdx.x) * 112)) + (yy_inner * 56)) + (((int)threadIdx.x) % 56))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
    }
  }
}


