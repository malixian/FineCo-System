
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
extern "C" __global__ void __launch_bounds__(392) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[128];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[(((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = data[((((rc_outer_outer * 6272) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 5488)];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (rc_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[rc_outer_inner]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 32)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 64)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 96)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 8)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 40)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 72)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 104)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 16)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 48)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 80)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 112)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 24)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 56)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 88)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[((rc_outer_inner * 392) + ((int)threadIdx.x))] * kernel_shared[(rc_outer_inner + 120)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    conv2d_nchw[(((((((int)blockIdx.x) >> 1) * 12544) + (ff_inner * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[((((((((int)blockIdx.x) >> 1) * 12544) + (ff_inner * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 3136)] = conv2d_nchw_local[(ff_inner + 4)];
    conv2d_nchw[((((((((int)blockIdx.x) >> 1) * 12544) + (ff_inner * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 6272)] = conv2d_nchw_local[(ff_inner + 8)];
    conv2d_nchw[((((((((int)blockIdx.x) >> 1) * 12544) + (ff_inner * 784)) + ((((int)blockIdx.x) & 1) * 392)) + ((int)threadIdx.x)) + 9408)] = conv2d_nchw_local[(ff_inner + 12)];
  }
}


