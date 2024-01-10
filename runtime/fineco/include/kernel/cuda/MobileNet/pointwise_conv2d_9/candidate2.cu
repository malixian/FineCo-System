
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
extern "C" __global__ void __launch_bounds__(196) candidate2(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[8];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[4096];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((rc_outer_outer * 6272) + ((int)threadIdx.x))];
    pad_temp_shared[(((int)threadIdx.x) + 196)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 196)];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 392)];
    pad_temp_shared[(((int)threadIdx.x) + 588)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 588)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 980)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 980)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1176)];
    pad_temp_shared[(((int)threadIdx.x) + 1372)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1372)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 1764)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1764)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 1960)];
    pad_temp_shared[(((int)threadIdx.x) + 2156)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2156)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 2548)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2548)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2744)];
    pad_temp_shared[(((int)threadIdx.x) + 2940)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 2940)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 3332)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3332)];
    pad_temp_shared[(((int)threadIdx.x) + 3528)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3528)];
    pad_temp_shared[(((int)threadIdx.x) + 3724)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3724)];
    pad_temp_shared[(((int)threadIdx.x) + 3920)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 4116)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4116)];
    pad_temp_shared[(((int)threadIdx.x) + 4312)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4312)];
    pad_temp_shared[(((int)threadIdx.x) + 4508)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4508)];
    pad_temp_shared[(((int)threadIdx.x) + 4704)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 4900)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 4900)];
    pad_temp_shared[(((int)threadIdx.x) + 5096)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5096)];
    pad_temp_shared[(((int)threadIdx.x) + 5292)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5292)];
    pad_temp_shared[(((int)threadIdx.x) + 5488)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5488)];
    pad_temp_shared[(((int)threadIdx.x) + 5684)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5684)];
    pad_temp_shared[(((int)threadIdx.x) + 5880)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 5880)];
    pad_temp_shared[(((int)threadIdx.x) + 6076)] = Input[(((rc_outer_outer * 6272) + ((int)threadIdx.x)) + 6076)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 7) * 1024)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) & 127))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 196) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 68) & 127))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 392) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 8) & 127))];
    kernel_shared[(((int)threadIdx.x) + 588)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 588) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 76) & 127))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 784) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 16) & 127))];
    kernel_shared[(((int)threadIdx.x) + 980)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 980) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 84) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1176)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1176) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 24) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1372)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1372) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 92) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1568) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1764)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1764) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 100) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1960)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 1960) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 40) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2156)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2156) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 108) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2352) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2548)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2548) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 116) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2744)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2744) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 56) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2940)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 2940) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 124) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3136) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3332)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3332) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 4) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3528)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3528) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 72) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3724)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3724) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 12) & 127))];
    if (((int)threadIdx.x) < 176) {
      kernel_shared[(((int)threadIdx.x) + 3920)] = kernel[((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) + 3920) >> 7) * 1024)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 128)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 256)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 384)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 512)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 640)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 768)]));
        conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 49)) + (((int)threadIdx.x) % 49))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 1024) + (rc_outer_inner * 32)) + rc_inner) + 896)]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    compute[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 49) * 392)) + (i1_inner * 49)) + (((int)threadIdx.x) % 49))] = max(conv2d_nchw[i1_inner], 0.000000e+00f);
  }
}


