#include "models/model.h"
#include "kernel/layer.h"
#include "../backend/cuda/cuda_timer.cc"
#include "../kernels/conv2d_relu.cpp"
#include "../kernels/maxpool.cpp"
#include "../kernels/dense.cpp"
#include "../kernels/depthwise_conv.cpp"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

class MobileNet final : public Model {
 public:
    
    MobileNet(const string& model_name = "MobileNet") : Model(model_name) {}
    
    void AddConv(int CO, int CI, int KH, int KW, int OH, int OW,  \
        const string& kernel_name, int N, int H, int W, bool need_input) {
        Conv2D* conv2d = new Conv2D(N, CI, H, W, CO, KH, KW, OH, OW, _runtime);
        // whether need input data
        if (need_input)
            conv2d->InitParams(true);
        else 
            conv2d->InitParams(false);
        ConfigureLayerCandidateByName(conv2d, _model_name, kernel_name);
    }

    void AddConvRelu(int CO, int CI, int KH, int KW, int OH, int OW, const string& kernel_name, int N, int H, int W, bool need_input) {
        Conv2DRelu* conv2d = new Conv2DRelu(N, CI, H, W, CO, KH, KW, OH, OW, _runtime);
        // whether need input data
        if (need_input)
            conv2d->InitParams(true);
        else 
            conv2d->InitParams(false);
        ConfigureLayerCandidateByName(conv2d, _model_name, kernel_name);
    }

    void AddDepthwiseConv(int CO, int CI, int KH, int KW, int OH, int OW, const string& kernel_name, int N, int H, int W, bool need_input) {
        DepthwiseConv* conv2d = new DepthwiseConv(N, CI, H, W, KH, KW, OH, OW);
        // whether need input data
        if (need_input)
            conv2d->InitParams(true);
        else 
            conv2d->InitParams(false);
        ConfigureLayerCandidateByName(conv2d, _model_name, kernel_name);
    }

    void AddMaxPool(int N, int C, int OH, int OW, const string& kernel_name) {
        MaxPool* maxpool = new MaxPool(N, C, OH, OW);
        maxpool->InitParams();
        ConfigureLayerCandidateByName(maxpool, _model_name, kernel_name);
    }

    void AddDense(int In, int Out, const string& kernel_name) {
        Dense* dense = new Dense(In, Out);
        dense->InitParams();
        ConfigureLayerCandidateByName(dense, _model_name, kernel_name);
    }
    
    void InitModel(RUNTIME runtime) final {
        /*====== AlexNet Layer 1 =======*/
        // Alloc Memory
        _runtime = runtime;

        // Input Size
        int N = 1, H = 224, W = 224, CO = 32, CI = 3, KH = 3, KW = 3, OH = H/2, OW = W/2;
        SetInPutSize(N*CI*H*W);
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv_1", N, H, W, true);
        
        CI = 32, CO = 1, KH = 3, KW = 3, H = OH, W = OW, OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW, "depthwise_conv2d_1",N, H, W, false);
        CI = 32, CO = 64, KH = 1, KW = 1, H = OH, W = OW,OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_1",N, H, W, false);


        CI = 64, CO = 1, KH = 3, KW = 3, H = OH/2, W = OW/2,OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW, "depthwise_conv2d_2", N, H, W, false);
        CI = 64, CO = 128, KH = 1, KW = 1, H = OH, W = OW,OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_2",N, H, W, false);
        CI = CO, CO = 1, KH = 3, KW = 3, H = OH, W = OW,OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW, "depthwise_conv2d_3", N, H, W, false);
        CI = 128, CO = 128, KH = 1, KW = 1, H = OH, W = OW,OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_3",N, H, W, false);

        CI = CO, CO = 1, KH = 3, KW = 3, H = OH/2, W = OW/2, OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW, "depthwise_conv2d_4", N, H, W, false);
        CI = 128, CO = 256, KH = 1, KW = 1, H = OH, W = OW,OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_4",N, H, W, false);
        CI = CO, CO = 1, KH = 3, KW = 3, H = OH, W = OW,OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW,"depthwise_conv2d_5", N, H, W, false);
        CI = 256, CO = 256, KH = 1, KW = 1, H = OH, W = OW,OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_5", N, H, W, false);

        CI = CO, CO = 1, KH = 3, KW = 3, H = OH/2, W = OW/2, OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW, "depthwise_conv2d_6",N, H, W, false);
        CI = 256, CO = 512, KH = 1, KW = 1, H = OH, W = OW,OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW,"pointwise_conv2d_6", N, H, W, false);

        for(int i=0; i<5; i++) {
            CI = CO, CO = 1, KH = 3, KW = 3, H = OH, W = OW, OH = H, OW = W;
            AddDepthwiseConv(CO, CI, KH, KW, OH, OW, "depthwise_conv2d_7", N, H, W, false);
            CI = 512, CO = 512, KH = 1, KW = 1, H = OH, W = OW, OH = H, OW = W;
            AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_7",N, H, W, false);
        }

        CI = CO, CO = 1, KH = 3, KW = 3, H = OH/2, W = OW/2,  OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW,"depthwise_conv2d_8", N, H, W, false);
        CI = 512, CO = 1024, KH = 1, KW = 1, H = OH, W = OW, OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "pointwise_conv2d_8", N, H, W, false);
        CI = CO, CO = 1, KH = 3, KW = 3, H = OH, W = OW, OH = H, OW = W;
        AddDepthwiseConv(CO, CI, KH, KW, OH, OW,"depthwise_conv2d_9", N, H, W, false);
        CI = 1024, CO = 1024, KH = 1, KW = 1, H = OH, W = OW, OH = H, OW = W;
        AddConvRelu(CO, CI, KH, KW, OH, OW,"pointwise_conv2d_9", N, H, W, false);

        
        CI = CO, OH = 1, OW = 1;
        AddMaxPool(N, CI, H, W, "maxpool");

        int In = N * CI * OH * OW;
        int Out = 1000;
        AddDense(In, Out, "dense");
        

        SetOutPutSize(Out);
       

        cout<<"Mobilnet Model Init Successfully"<<endl;

    }


};