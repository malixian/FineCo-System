
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
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[6960];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((int)threadIdx.x) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) % 15)))) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((int)threadIdx.x) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 196)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 1) % 15))) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + (((((int)threadIdx.x) + 196) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 392)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 392) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 2) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 392) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 392) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 588)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 3) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 588) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 153) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 784)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 349) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 4) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 784) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 349) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 980)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 5) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 980) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 110) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 306) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 6) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1176) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 306) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1372)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 7) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1372) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 67) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 263) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 8) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1568) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 263) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1764)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 9) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1764) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 24) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 10) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1960) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 220) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2156)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 416) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 11) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2156) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 416) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 12) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2352) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 177) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2548)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 373) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 13) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2548) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 373) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 14) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2744) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 134) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2940)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) / 15) + 22) % 29))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2940) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) / 15) + 22) % 29) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 1) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 3136) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 91) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3332)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 287) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 2) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 3332) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 287) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3528)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 3) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 3528) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 48) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3724)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 244) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 4) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 3724) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 244) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3920)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 5) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 5) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 3920) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 5) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 4116)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 6) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 4116) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 201) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 6) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 4312)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 397) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 7) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 4312) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 397) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 7) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 4508)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 8) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 4508) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 158) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 8) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 4704)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 354) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 9) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 4704) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 354) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 9) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 4900)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 10) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 4900) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 115) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 10) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 5096)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 311) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 11) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 5096) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 311) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 11) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 5292)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 12) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 5292) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 72) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 12) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 5488)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 268) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 13) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 5488) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 268) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 13) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 5684)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 14) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 5684) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 29) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 14) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 5880)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + (((int)threadIdx.x) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 5880) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + (((((int)threadIdx.x) / 15) + 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + (((int)threadIdx.x) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 6076)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 421) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 1) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 6076) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 421) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 1) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 6272)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 2) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 6272) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 182) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 2) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 6468)] = (((1 <= ((((((int)blockIdx.x) & 7) >> 2) * 28) + (((((int)threadIdx.x) + 378) % 435) / 15))) && (1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 3) % 15)))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 6468) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 378) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 3) % 15)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 6664)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 4) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 6664) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 139) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 4) % 15)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 100) {
      pad_temp_shared[(((int)threadIdx.x) + 6860)] = ((1 <= (((((int)blockIdx.x) & 3) * 14) + ((((int)threadIdx.x) + 5) % 15))) ? data[(((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 6860) / 435) * 3136)) + (((((int)blockIdx.x) & 7) >> 2) * 1568)) + ((((((int)threadIdx.x) + 335) % 435) / 15) * 56)) + ((((int)blockIdx.x) & 3) * 14)) + ((((int)threadIdx.x) + 5) % 15)) - 57)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + ((((int)threadIdx.x) / 144) * 1152)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 196) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 52) % 144))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 392) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 104) % 144))];
    kernel_shared[(((int)threadIdx.x) + 588)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 588) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 12) % 144))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 784) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    kernel_shared[(((int)threadIdx.x) + 980)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 980) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 116) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1176)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1176) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 24) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1372)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1372) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 76) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1568) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 128) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1764)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1764) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 36) % 144))];
    kernel_shared[(((int)threadIdx.x) + 1960)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 1960) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 88) % 144))];
    if (((int)threadIdx.x) < 148) {
      kernel_shared[(((int)threadIdx.x) + 2156)] = kernel[(((((((int)blockIdx.x) >> 3) * 18432) + (((((int)threadIdx.x) + 2156) / 144) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 140) % 144))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3))]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1152)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 144)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1296)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 288)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1440)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 432)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1584)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1153)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 145)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1297)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 289)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1441)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 433)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1585)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 2)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1154)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 146)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1298)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 290)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1442)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 434)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((((rc_outer_inner * 3480) + (rc_inner * 435)) + (((((int)threadIdx.x) % 98) / 7) * 30)) + (ry_inner * 15)) + ((((int)threadIdx.x) % 7) * 2)) + 2)] * kernel_shared[((((((((int)threadIdx.x) / 98) * 576) + (rc_outer_inner * 72)) + (rc_inner * 9)) + (ry_inner * 3)) + 1586)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    conv2d_nchw[((((((((((int)blockIdx.x) >> 3) * 12544) + ((((int)threadIdx.x) / 98) * 3136)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[(((((((((((int)blockIdx.x) >> 3) * 12544) + ((((int)threadIdx.x) / 98) * 3136)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 7) >> 2) * 392)) + (((((int)threadIdx.x) % 98) / 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7)) + 6272)] = conv2d_nchw_local[(ff_inner + 4)];
  }
}


