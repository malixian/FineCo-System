

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
extern "C" __global__ void __launch_bounds__(121) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[12];
  __shared__ float pad_temp_shared[2601];
  __shared__ float kernel_shared[1452];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 3; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= ((((((int)blockIdx.x) % 25) / 5) * 22) + (((int)threadIdx.x) / 102))) && (1 <= (((((int)blockIdx.x) % 5) * 22) + ((((int)threadIdx.x) % 51) >> 1)))) && ((((((int)blockIdx.x) % 5) * 22) + ((((int)threadIdx.x) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + ((((int)threadIdx.x) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + (((int)threadIdx.x) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 121)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 19) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 19) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 121) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 19) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 242)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 38) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 38) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 242) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 38) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 363)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 6) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 6) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 363) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 6) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 484)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 25) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 25) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 484) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 25) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 605)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 44) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 44) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 605) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 44) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 726)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 12) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 12) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 726) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 12) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 847)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 31) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 31) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 847) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 31) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 968)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 50) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 50) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 968) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 50) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1089)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 18) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 18) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1089) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 18) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1210)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 37) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 37) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1210) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 37) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1331)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 5) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 5) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1331) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 5) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1452)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 24) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 24) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1452) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 24) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1573)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 43) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 43) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1573) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 43) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1694)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 11) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 11) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1694) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 11) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1815)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 30) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 30) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1815) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 30) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1936)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 49) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 49) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 1936) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 49) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2057)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 17) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 17) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 2057) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 17) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2178)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 36) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 36) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 2178) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 36) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2299)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 4) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 4) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 2299) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 4) % 51)) - 450)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2420)] = (((1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 23) % 51) >> 1))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 23) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 2420) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 23) % 51)) - 450)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 60) {
      pad_temp_shared[(((int)threadIdx.x) + 2541)] = ((((((((((int)blockIdx.x) % 25) / 5) * 22) + ((((int)threadIdx.x) + 2541) / 102)) < 113) && (1 <= (((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 42) % 51) >> 1)))) && ((((((int)blockIdx.x) % 5) * 22) + (((((int)threadIdx.x) + 42) % 51) >> 1)) < 113)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 25) / 5) * 9856)) + (((((int)threadIdx.x) + 2541) / 51) * 224)) + ((((int)blockIdx.x) % 5) * 44)) + ((((int)threadIdx.x) + 42) % 51)) - 450)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 121)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 363)];
    kernel_shared[(((int)threadIdx.x) + 242)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 726)];
    kernel_shared[(((int)threadIdx.x) + 363)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 1089)];
    kernel_shared[(((int)threadIdx.x) + 484)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 1452)];
    kernel_shared[(((int)threadIdx.x) + 605)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 1815)];
    kernel_shared[(((int)threadIdx.x) + 726)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 2178)];
    kernel_shared[(((int)threadIdx.x) + 847)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 2541)];
    kernel_shared[(((int)threadIdx.x) + 968)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 2904)];
    kernel_shared[(((int)threadIdx.x) + 1089)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 3267)];
    kernel_shared[(((int)threadIdx.x) + 1210)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 3630)];
    kernel_shared[(((int)threadIdx.x) + 1331)] = kernel[(((((((int)blockIdx.x) / 25) * 4356) + (rc_outer_outer * 121)) + ((int)threadIdx.x)) + 3993)];
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 11; ++ry_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[(ry_inner * 11)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 484)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 968)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 121)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 605)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 1089)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 242)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 726)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 1210)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 363)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 847)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4))] * kernel_shared[((ry_inner * 11) + 1331)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 1)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 485)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 969)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 122)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 606)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 1090)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 243)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 727)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 1211)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 364)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 848)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 1)] * kernel_shared[((ry_inner * 11) + 1332)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 2)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 486)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 970)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 123)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 607)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 1091)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 244)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 728)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 1212)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 365)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 849)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 2)] * kernel_shared[((ry_inner * 11) + 1333)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 3)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 487)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 971)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 124)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 608)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 1092)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 245)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 729)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 1213)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 366)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 850)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 3)] * kernel_shared[((ry_inner * 11) + 1334)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 4)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 488)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 972)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 125)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 609)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 1093)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 246)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 730)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 1214)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 367)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 851)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 4)] * kernel_shared[((ry_inner * 11) + 1335)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 5)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 489)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 973)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 126)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 610)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 1094)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 247)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 731)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 1215)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 368)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 852)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 5)] * kernel_shared[((ry_inner * 11) + 1336)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 6)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 490)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 974)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 127)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 611)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 1095)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 248)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 732)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 1216)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 369)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 853)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 6)] * kernel_shared[((ry_inner * 11) + 1337)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 7)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 491)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 975)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 128)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 612)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 1096)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 249)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 733)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 1217)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 370)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 854)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 7)] * kernel_shared[((ry_inner * 11) + 1338)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 8)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 492)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 976)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 129)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 613)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 1097)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 250)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 734)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 1218)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 371)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 855)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 8)] * kernel_shared[((ry_inner * 11) + 1339)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 9)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 493)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 977)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 130)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 614)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 1098)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 251)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 735)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 1219)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 372)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 856)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 9)] * kernel_shared[((ry_inner * 11) + 1340)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 10)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 494)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 978)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 131)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 615)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 1099)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 252)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 736)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 1220)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 373)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 857)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((((((int)threadIdx.x) / 11) * 204) + (ry_inner * 51)) + ((((int)threadIdx.x) % 11) * 4)) + 10)] * kernel_shared[((ry_inner * 11) + 1341)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    compute[(((((((((int)blockIdx.x) / 25) * 36300) + (i1_inner * 3025)) + (((((int)blockIdx.x) % 25) / 5) * 605)) + ((((int)threadIdx.x) / 11) * 55)) + ((((int)blockIdx.x) % 5) * 11)) + (((int)threadIdx.x) % 11))] = max((conv2d_nchw[i1_inner] + bias[(((((int)blockIdx.x) / 25) * 12) + i1_inner)]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) / 25) * 36300) + (i1_inner * 3025)) + (((((int)blockIdx.x) % 25) / 5) * 605)) + ((((int)threadIdx.x) / 11) * 55)) + ((((int)blockIdx.x) % 5) * 11)) + (((int)threadIdx.x) % 11)) + 12100)] = max((conv2d_nchw[(i1_inner + 4)] + bias[((((((int)blockIdx.x) / 25) * 12) + i1_inner) + 4)]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.x) / 25) * 36300) + (i1_inner * 3025)) + (((((int)blockIdx.x) % 25) / 5) * 605)) + ((((int)threadIdx.x) / 11) * 55)) + ((((int)blockIdx.x) % 5) * 11)) + (((int)threadIdx.x) % 11)) + 24200)] = max((conv2d_nchw[(i1_inner + 8)] + bias[((((((int)blockIdx.x) / 25) * 12) + i1_inner) + 8)]), 0.000000e+00f);
  }
}


