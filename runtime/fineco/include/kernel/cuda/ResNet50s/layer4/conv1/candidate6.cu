
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
extern "C" __global__ void __launch_bounds__(196) candidate6(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[256];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[(((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 196)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 588)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 980)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 1372)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 5488)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1764)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 7056)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 2156)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 8624)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 2548)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 10192)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 10976)];
    pad_temp_shared[(((int)threadIdx.x) + 2940)] = data[((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 11760)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    if (((int)threadIdx.x) < 60) {
      kernel_shared[(((int)threadIdx.x) + 196)] = kernel[(((((((int)blockIdx.x) >> 2) * 8192) + (((((int)threadIdx.x) + 196) >> 4) * 512)) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) + 4) & 15))];
    }
    __syncthreads();
    for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
      for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[((rc_inner * 196) + ((int)threadIdx.x))] * kernel_shared[((ff_c_outer_inner * 32) + rc_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 8)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 8)] + (pad_temp_shared[((rc_inner * 196) + ((int)threadIdx.x))] * kernel_shared[(((ff_c_outer_inner * 32) + rc_inner) + 128)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[((rc_inner * 196) + ((int)threadIdx.x))] * kernel_shared[(((ff_c_outer_inner * 32) + rc_inner) + 16)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 9)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 9)] + (pad_temp_shared[((rc_inner * 196) + ((int)threadIdx.x))] * kernel_shared[(((ff_c_outer_inner * 32) + rc_inner) + 144)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    conv2d_nchw[(((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[((((((((int)blockIdx.x) >> 2) * 12544) + (ff_inner * 784)) + ((((int)blockIdx.x) & 3) * 196)) + ((int)threadIdx.x)) + 6272)] = conv2d_nchw_local[(ff_inner + 8)];
  }
}


