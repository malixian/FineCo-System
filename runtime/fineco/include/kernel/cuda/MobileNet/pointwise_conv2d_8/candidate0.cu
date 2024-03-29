
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
extern "C" __global__ void __launch_bounds__(49) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[4];
  __shared__ float pad_temp_shared[3136];
  __shared__ float kernel_shared[256];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    *(float4*)(pad_temp_shared + (((int)threadIdx.x) * 4)) = *(float4*)(Input + ((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 196)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 196));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 392)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 392));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 588)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 588));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 784)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 784));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 980)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 980));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 1176)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 1176));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 1372)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 1372));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 1568)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 1568));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 1764)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 1764));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 1960)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 1960));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 2156)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 2156));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 2352)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 2352));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 2548)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 2548));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 2744)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 2744));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 2940)) = *(float4*)(Input + (((rc_outer_outer * 3136) + (((int)threadIdx.x) * 4)) + 2940));
    kernel_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) * 2048) + (rc_outer_outer * 64)) + ((int)threadIdx.x))];
    kernel_shared[(((int)threadIdx.x) + 49)] = kernel[((((((int)blockIdx.x) * 2048) + (((((int)threadIdx.x) + 49) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 49) & 63))];
    kernel_shared[(((int)threadIdx.x) + 98)] = kernel[((((((int)blockIdx.x) * 2048) + (((((int)threadIdx.x) + 98) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 34) & 63))];
    kernel_shared[(((int)threadIdx.x) + 147)] = kernel[((((((int)blockIdx.x) * 2048) + (((((int)threadIdx.x) + 147) >> 6) * 512)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 19) & 63))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[((((((int)blockIdx.x) * 2048) + (((((int)threadIdx.x) + 196) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 4))];
    if (((int)threadIdx.x) < 11) {
      kernel_shared[(((int)threadIdx.x) + 245)] = kernel[((((((int)blockIdx.x) * 2048) + (((((int)threadIdx.x) + 245) >> 6) * 512)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) + 53))];
    }
    __syncthreads();
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[0]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[64]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[128]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[192]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 49)] * kernel_shared[1]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 49)] * kernel_shared[65]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 49)] * kernel_shared[129]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 49)] * kernel_shared[193]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 98)] * kernel_shared[2]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 98)] * kernel_shared[66]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 98)] * kernel_shared[130]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 98)] * kernel_shared[194]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 147)] * kernel_shared[3]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 147)] * kernel_shared[67]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 147)] * kernel_shared[131]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 147)] * kernel_shared[195]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 196)] * kernel_shared[4]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 196)] * kernel_shared[68]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 196)] * kernel_shared[132]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 196)] * kernel_shared[196]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 245)] * kernel_shared[5]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 245)] * kernel_shared[69]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 245)] * kernel_shared[133]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 245)] * kernel_shared[197]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 294)] * kernel_shared[6]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 294)] * kernel_shared[70]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 294)] * kernel_shared[134]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 294)] * kernel_shared[198]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 343)] * kernel_shared[7]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 343)] * kernel_shared[71]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 343)] * kernel_shared[135]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 343)] * kernel_shared[199]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 392)] * kernel_shared[8]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 392)] * kernel_shared[72]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 392)] * kernel_shared[136]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 392)] * kernel_shared[200]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 441)] * kernel_shared[9]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 441)] * kernel_shared[73]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 441)] * kernel_shared[137]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 441)] * kernel_shared[201]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 490)] * kernel_shared[10]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 490)] * kernel_shared[74]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 490)] * kernel_shared[138]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 490)] * kernel_shared[202]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 539)] * kernel_shared[11]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 539)] * kernel_shared[75]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 539)] * kernel_shared[139]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 539)] * kernel_shared[203]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 588)] * kernel_shared[12]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 588)] * kernel_shared[76]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 588)] * kernel_shared[140]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 588)] * kernel_shared[204]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 637)] * kernel_shared[13]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 637)] * kernel_shared[77]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 637)] * kernel_shared[141]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 637)] * kernel_shared[205]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 686)] * kernel_shared[14]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 686)] * kernel_shared[78]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 686)] * kernel_shared[142]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 686)] * kernel_shared[206]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 735)] * kernel_shared[15]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 735)] * kernel_shared[79]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 735)] * kernel_shared[143]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 735)] * kernel_shared[207]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 784)] * kernel_shared[16]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 784)] * kernel_shared[80]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 784)] * kernel_shared[144]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 784)] * kernel_shared[208]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 833)] * kernel_shared[17]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 833)] * kernel_shared[81]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 833)] * kernel_shared[145]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 833)] * kernel_shared[209]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 882)] * kernel_shared[18]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 882)] * kernel_shared[82]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 882)] * kernel_shared[146]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 882)] * kernel_shared[210]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 931)] * kernel_shared[19]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 931)] * kernel_shared[83]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 931)] * kernel_shared[147]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 931)] * kernel_shared[211]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 980)] * kernel_shared[20]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 980)] * kernel_shared[84]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 980)] * kernel_shared[148]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 980)] * kernel_shared[212]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1029)] * kernel_shared[21]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1029)] * kernel_shared[85]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1029)] * kernel_shared[149]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1029)] * kernel_shared[213]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1078)] * kernel_shared[22]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1078)] * kernel_shared[86]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1078)] * kernel_shared[150]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1078)] * kernel_shared[214]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1127)] * kernel_shared[23]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1127)] * kernel_shared[87]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1127)] * kernel_shared[151]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1127)] * kernel_shared[215]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1176)] * kernel_shared[24]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1176)] * kernel_shared[88]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1176)] * kernel_shared[152]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1176)] * kernel_shared[216]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1225)] * kernel_shared[25]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1225)] * kernel_shared[89]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1225)] * kernel_shared[153]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1225)] * kernel_shared[217]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1274)] * kernel_shared[26]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1274)] * kernel_shared[90]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1274)] * kernel_shared[154]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1274)] * kernel_shared[218]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1323)] * kernel_shared[27]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1323)] * kernel_shared[91]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1323)] * kernel_shared[155]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1323)] * kernel_shared[219]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1372)] * kernel_shared[28]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1372)] * kernel_shared[92]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1372)] * kernel_shared[156]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1372)] * kernel_shared[220]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1421)] * kernel_shared[29]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1421)] * kernel_shared[93]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1421)] * kernel_shared[157]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1421)] * kernel_shared[221]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1470)] * kernel_shared[30]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1470)] * kernel_shared[94]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1470)] * kernel_shared[158]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1470)] * kernel_shared[222]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1519)] * kernel_shared[31]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1519)] * kernel_shared[95]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1519)] * kernel_shared[159]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1519)] * kernel_shared[223]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1568)] * kernel_shared[32]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1568)] * kernel_shared[96]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1568)] * kernel_shared[160]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1568)] * kernel_shared[224]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1617)] * kernel_shared[33]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1617)] * kernel_shared[97]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1617)] * kernel_shared[161]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1617)] * kernel_shared[225]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1666)] * kernel_shared[34]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1666)] * kernel_shared[98]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1666)] * kernel_shared[162]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1666)] * kernel_shared[226]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1715)] * kernel_shared[35]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1715)] * kernel_shared[99]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1715)] * kernel_shared[163]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1715)] * kernel_shared[227]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1764)] * kernel_shared[36]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1764)] * kernel_shared[100]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1764)] * kernel_shared[164]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1764)] * kernel_shared[228]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1813)] * kernel_shared[37]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1813)] * kernel_shared[101]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1813)] * kernel_shared[165]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1813)] * kernel_shared[229]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1862)] * kernel_shared[38]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1862)] * kernel_shared[102]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1862)] * kernel_shared[166]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1862)] * kernel_shared[230]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1911)] * kernel_shared[39]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1911)] * kernel_shared[103]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1911)] * kernel_shared[167]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1911)] * kernel_shared[231]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 1960)] * kernel_shared[40]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 1960)] * kernel_shared[104]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 1960)] * kernel_shared[168]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 1960)] * kernel_shared[232]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2009)] * kernel_shared[41]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2009)] * kernel_shared[105]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2009)] * kernel_shared[169]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2009)] * kernel_shared[233]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2058)] * kernel_shared[42]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2058)] * kernel_shared[106]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2058)] * kernel_shared[170]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2058)] * kernel_shared[234]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2107)] * kernel_shared[43]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2107)] * kernel_shared[107]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2107)] * kernel_shared[171]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2107)] * kernel_shared[235]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2156)] * kernel_shared[44]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2156)] * kernel_shared[108]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2156)] * kernel_shared[172]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2156)] * kernel_shared[236]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2205)] * kernel_shared[45]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2205)] * kernel_shared[109]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2205)] * kernel_shared[173]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2205)] * kernel_shared[237]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2254)] * kernel_shared[46]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2254)] * kernel_shared[110]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2254)] * kernel_shared[174]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2254)] * kernel_shared[238]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2303)] * kernel_shared[47]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2303)] * kernel_shared[111]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2303)] * kernel_shared[175]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2303)] * kernel_shared[239]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2352)] * kernel_shared[48]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2352)] * kernel_shared[112]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2352)] * kernel_shared[176]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2352)] * kernel_shared[240]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2401)] * kernel_shared[49]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2401)] * kernel_shared[113]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2401)] * kernel_shared[177]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2401)] * kernel_shared[241]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2450)] * kernel_shared[50]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2450)] * kernel_shared[114]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2450)] * kernel_shared[178]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2450)] * kernel_shared[242]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2499)] * kernel_shared[51]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2499)] * kernel_shared[115]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2499)] * kernel_shared[179]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2499)] * kernel_shared[243]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2548)] * kernel_shared[52]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2548)] * kernel_shared[116]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2548)] * kernel_shared[180]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2548)] * kernel_shared[244]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2597)] * kernel_shared[53]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2597)] * kernel_shared[117]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2597)] * kernel_shared[181]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2597)] * kernel_shared[245]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2646)] * kernel_shared[54]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2646)] * kernel_shared[118]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2646)] * kernel_shared[182]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2646)] * kernel_shared[246]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2695)] * kernel_shared[55]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2695)] * kernel_shared[119]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2695)] * kernel_shared[183]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2695)] * kernel_shared[247]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2744)] * kernel_shared[56]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2744)] * kernel_shared[120]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2744)] * kernel_shared[184]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2744)] * kernel_shared[248]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2793)] * kernel_shared[57]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2793)] * kernel_shared[121]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2793)] * kernel_shared[185]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2793)] * kernel_shared[249]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2842)] * kernel_shared[58]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2842)] * kernel_shared[122]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2842)] * kernel_shared[186]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2842)] * kernel_shared[250]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2891)] * kernel_shared[59]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2891)] * kernel_shared[123]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2891)] * kernel_shared[187]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2891)] * kernel_shared[251]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2940)] * kernel_shared[60]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2940)] * kernel_shared[124]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2940)] * kernel_shared[188]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2940)] * kernel_shared[252]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 2989)] * kernel_shared[61]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 2989)] * kernel_shared[125]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 2989)] * kernel_shared[189]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 2989)] * kernel_shared[253]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 3038)] * kernel_shared[62]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 3038)] * kernel_shared[126]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 3038)] * kernel_shared[190]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 3038)] * kernel_shared[254]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) + 3087)] * kernel_shared[63]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) + 3087)] * kernel_shared[127]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) + 3087)] * kernel_shared[191]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) + 3087)] * kernel_shared[255]));
  }
  compute[((((int)blockIdx.x) * 196) + ((int)threadIdx.x))] = max(conv2d_nchw[0], 0.000000e+00f);
  compute[(((((int)blockIdx.x) * 196) + ((int)threadIdx.x)) + 49)] = max(conv2d_nchw[1], 0.000000e+00f);
  compute[(((((int)blockIdx.x) * 196) + ((int)threadIdx.x)) + 98)] = max(conv2d_nchw[2], 0.000000e+00f);
  compute[(((((int)blockIdx.x) * 196) + ((int)threadIdx.x)) + 147)] = max(conv2d_nchw[3], 0.000000e+00f);
}


