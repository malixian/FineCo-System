
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
extern "C" __global__ void __launch_bounds__(112) candidate4(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[3584];
  __shared__ float kernel_shared[8192];
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
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 112)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 336)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 560)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 5488)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 7056)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 8624)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 1456)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 10192)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 10976)];
    pad_temp_shared[(((int)threadIdx.x) + 1680)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 11760)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 1904)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 13328)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 14112)];
    pad_temp_shared[(((int)threadIdx.x) + 2128)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 14896)];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 15680)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 16464)];
    pad_temp_shared[(((int)threadIdx.x) + 2464)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 17248)];
    pad_temp_shared[(((int)threadIdx.x) + 2576)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 18032)];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 18816)];
    pad_temp_shared[(((int)threadIdx.x) + 2800)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 19600)];
    pad_temp_shared[(((int)threadIdx.x) + 2912)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 20384)];
    pad_temp_shared[(((int)threadIdx.x) + 3024)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 21168)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 21952)];
    pad_temp_shared[(((int)threadIdx.x) + 3248)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 22736)];
    pad_temp_shared[(((int)threadIdx.x) + 3360)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 23520)];
    pad_temp_shared[(((int)threadIdx.x) + 3472)] = Input[(((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 24304)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 112) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 224) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 336) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 448) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 560) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 672) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 784) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1008) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1120) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1232) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1344) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1456) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1568) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1680) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 1904) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2016) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2128) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2240) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2352) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2464) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2576)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2576) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 5376)];
    kernel_shared[(((int)threadIdx.x) + 2800)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2800) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 2912) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3024)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3024) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3136) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3248)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3248) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3360) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3472)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3472) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 3696)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3696) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3808) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 3920)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 3920) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4032) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4144)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4144) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4256)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4256) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4368)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4368) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 8960)];
    kernel_shared[(((int)threadIdx.x) + 4592)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4592) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4704)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4704) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4816)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4816) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 4928)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 4928) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5040)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5040) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5152)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5152) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5264)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5264) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 5488)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5488) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5600)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5600) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5712)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5712) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5824)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5824) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 5936)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 5936) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6048)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6048) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6160)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6160) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 6384)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6384) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6496)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6496) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6608)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6608) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6720)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6720) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6832)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6832) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 6944)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 6944) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7056)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7056) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 7280)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7280) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 112) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7392)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7392) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 96) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7504)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7504) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 80) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7616)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7616) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 64) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7728)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7728) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 48) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7840)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7840) >> 7) * 256)) + (rc_outer_outer * 128)) + ((((int)threadIdx.x) + 32) & 127))];
    kernel_shared[(((int)threadIdx.x) + 7952)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (((((int)threadIdx.x) + 7952) >> 7) * 256)) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 16))];
    kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + ((int)threadIdx.x)) + 16128)];
    if (((int)threadIdx.x) < 16) {
      kernel_shared[(((int)threadIdx.x) + 8176)] = kernel[(((((((int)blockIdx.x) / 7) * 16384) + (rc_outer_outer * 128)) + (((int)threadIdx.x) + 112)) + 16128)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 64; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2))]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2))]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 128)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 128)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 256)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 256)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 384)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 384)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 129)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 129)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 257)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 257)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 385)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 385)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 640)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 640)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 768)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 768)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((rc_outer_inner * 56) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 896)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 896)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 641)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 641)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 769)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 769)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 897)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 1024) + (rc_outer_inner * 2)) + 897)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    compute[((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 1568)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14))] = max(conv2d_nchw[i1_inner], 0.000000e+00f);
    compute[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) / 14) * 1568)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 14)) + 14)] = max(conv2d_nchw[(i1_inner + 8)], 0.000000e+00f);
  }
}


