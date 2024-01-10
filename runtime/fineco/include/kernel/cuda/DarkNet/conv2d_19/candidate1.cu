
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
extern "C" __global__ void __launch_bounds__(196) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[10];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[2560];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((rc_outer_outer * 3136) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 196)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 196)];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 392)];
    pad_temp_shared[(((int)threadIdx.x) + 588)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 588)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 980)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 980)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1176)];
    pad_temp_shared[(((int)threadIdx.x) + 1372)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1372)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 1764)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1764)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 1960)];
    pad_temp_shared[(((int)threadIdx.x) + 2156)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2156)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 2548)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2548)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2744)];
    pad_temp_shared[(((int)threadIdx.x) + 2940)] = data[(((rc_outer_outer * 3136) + ((int)threadIdx.x)) + 2940)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 40960) + ((((int)threadIdx.x) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 196) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 4) & 63))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 392) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 8) & 63))];
    kernel_shared[(((int)threadIdx.x) + 588)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 588) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 12) & 63))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 784) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 16) & 63))];
    kernel_shared[(((int)threadIdx.x) + 980)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 980) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 20) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1176)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 1176) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 24) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1372)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 1372) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 28) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 1568) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1764)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 1764) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 36) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1960)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 1960) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 40) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2156)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 2156) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 44) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 2352) >> 6) * 1024)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 48) & 63))];
    if (((int)threadIdx.x) < 12) {
      kernel_shared[(((int)threadIdx.x) + 2548)] = kernel[((((((int)blockIdx.x) * 40960) + (((((int)threadIdx.x) + 2548) >> 6) * 1024)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 52))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 64)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 128)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 192)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 256)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 320)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 384)]));
        conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 448)]));
        conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 512)]));
        conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 784) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 640) + (rc_outer_inner * 16)) + rc_inner) + 576)]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 10; ++i1_inner) {
    compute[((((((int)blockIdx.x) * 1960) + ((((int)threadIdx.x) / 49) * 490)) + (i1_inner * 49)) + (((int)threadIdx.x) % 49))] = max((conv2d_nchw[i1_inner] + bias[(((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) / 49) * 10)) + i1_inner)]), 0.000000e+00f);
  }
}


