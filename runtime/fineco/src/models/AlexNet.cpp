#ifndef AlexNet_H_
#define AlexNet_H_

#include "models/model.h"
#include "kernel/layer.h"
#include "kernel/sleep.h"
#include "../backend/cuda/cuda_timer.cc"
#include "../kernels/conv2d_relu.cpp"
#include "../kernels/maxpool.cpp"
#include "../kernels/dense.cpp"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

class AlexNet final : public Model {
 public:
    
    AlexNet(const string& model_name = "AlexNet") : Model(model_name) {}
    
    void AddConvRelu(int CO, int CI, int KH, int KW, int OH, int OW, int N,  int H, int W, bool need_input) {
        Conv2DRelu* conv2d = new Conv2DRelu(N, CI, H, W, CO, KH, KW, OH, OW, _runtime);
        // whether need input data
        if (need_input)
            conv2d->InitParams(true);
        else 
            conv2d->InitParams(false);
        ConfigureLayerCandidate(conv2d);
    }

    void AddMaxPool(int N, int C, int OH, int OW) {
        MaxPool* maxpool = new MaxPool(N, C, OH, OW);
        maxpool->InitParams();
        ConfigureLayerCandidate(maxpool);
    }

    void AddDense(int In, int Out) {
        Dense* dense = new Dense(In, Out);
        dense->InitParams();
        ConfigureLayerCandidate(dense);
    }
    
    void InitModel(RUNTIME runtime) final {
        /*====== AlexNet Layer 1 =======*/
        // Alloc Memory
        _runtime = runtime;

        // Input Size
        int N = 1, H = 224, W = 224, CO = 48, CI = 3, KH = 11, KW = 11, OH = 55, OW = 55;
        AddConvRelu(CO, CI, KH, KW, OH, OW, N, H, W, true);
        SetInPutSize(N*CI*H*W);
        
        CI = CO, OH = 27, OW = 27, H = 27, W = 27;
        AddMaxPool(N, CI, OH, OW);

        CI = CO, CO = 128, KH = 5, KW = 5, OH = 27, OW = 27;
        AddConvRelu(CO, CI, KH, KW, OH, OW, N, H, W, false);

        CI = CO, OH = 13, OW = 13, H = 13, W = 13;
        AddMaxPool(N, CI, OH, OW);

        CI = CO, CO = 192, KH = 3, KW = 3, OH = 13, OW = 13;
        AddConvRelu(CO, CI, KH, KW, OH, OW, N, H, W, false);

        CI = CO, CO = 192, KH = 3, KW = 3, OH = 13, OW = 13;
        AddConvRelu(CO, CI, KH, KW, OH, OW, N, H, W, false);

        CI = CO, CO = 128, KH = 3, KW = 3, OH = 13, OW = 13;
        AddConvRelu(CO, CI, KH, KW, OH, OW, N, H, W, false);

        CI = CO, OH = 6, OW = 6;
        AddMaxPool(N, CI, OH, OW);

        int In = N * CI * OH * OW;
        int Out = 2048;
        AddDense(In, Out);

        In = 2048, Out = 2048;
        AddDense(In, Out);

        In = 2048, Out = 500;
        AddDense(In, Out);
        
        SetOutPutSize(Out);

        cout<<"Alexnet Model Init Successfully"<<endl;

    }


};

#endif