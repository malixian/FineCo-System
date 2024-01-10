#include "models/model.h"
#include "kernel/layer.h"
#include "../backend/cuda/cuda_timer.cc"
#include "../kernels/conv2d_relu.cpp"
#include "../kernels/maxpool.cpp"
#include "../kernels/dense.cpp"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

class DarkNet final : public Model {
 public:
    
    DarkNet(const string& model_name = "DarkNet") : Model(model_name) {}
    
    void AddConvRelu(int CO, int CI, int KH, int KW, int OH, int OW, const string& kernel_name, int N, int H, int W, bool need_input) {
        Conv2DRelu* conv2d = new Conv2DRelu(N, CI, H, W, CO, KH, KW, OH, OW, _runtime);
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
    
    void InitModel(RUNTIME runtime) final {
        /*====== AlexNet Layer 1 =======*/
        // Alloc Memory
        _runtime = runtime;

        //AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);

        // Input Size
        int N = 1, H = 224, W = 224, CO = 32, CI = 3, KH = 3, KW = 3, OH = H, OW = W;
        SetInPutSize(N*CI*H*W);
        // 1
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_1",  N, H, W, true);
        
        CI = CO, OH = H/2, OW = W/2, H = OH, W = OW;
        AddMaxPool(N, CI, OH, OW, "maxpool_1");

        // 2
        CI = CO, CO = 64, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_2", N, H, W, false);

        CI = CO, OH = H/2, OW = W/2, H = OH, W = OW;
        AddMaxPool(N, CI, OH, OW, "maxpool_2");

        // 3
        CI = CO, CO = 128, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_3",N, H, W, false);

        // 4
        CI = CO, CO = 64, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW,"conv2d_4", N, H, W, false);

        // 5
        CI = CO, CO = 128, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_3", N, H, W, false);

        
        CI = CO, OH = H/2, OW = W/2, H = OH, W = OW;
        AddMaxPool(N, CI, OH, OW, "maxpool_3");
        // 6
        CI = CO, CO = 256, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW,"conv2d_6", N, H, W, false);

        CI = CO, CO = 128, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW,"conv2d_7", N, H, W, false);

        CI = CO, CO = 256, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_6", N, H, W, false);

        CI = CO, OH = H/2, OW = W/2, H = OH, W = OW;
        AddMaxPool(N, CI, OH, OW, "maxpool_4");

        CI = CO, CO = 512, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_9", N, H, W, false);

        CI = CO, CO = 256, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_10",N, H, W, false);

        CI = CO, CO = 512, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_9",N, H, W, false);

        CI = CO, CO = 256, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_10",N, H, W, false);

        CI = CO, CO = 512, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW,"conv2d_9", N, H, W, false);

        CI = CO, OH = H/2, OW = W/2, H = OH, W = OW;
        AddMaxPool(N, CI, OH, OW, "maxpool_5");

        CI = CO, CO = 1024, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_14", N, H, W, false);

        CI = CO, CO = 512, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_15",N, H, W, false);

        CI = CO, CO = 1024, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_14",N, H, W, false);

        CI = CO, CO = 512, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_15",N, H, W, false);

        CI = CO, CO = 1024, KH = 3, KW = 3, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_14",N, H, W, false);

        CI = CO, CO = 1000, KH = 1, KW = 1, H = OH, W = OW;;
        AddConvRelu(CO, CI, KH, KW, OH, OW, "conv2d_15",N, H, W, false);

        SetOutPutSize(N * CO * OH * OW);

        cout<<"DarkNet Model Init Successfully"<<endl;

    }


};