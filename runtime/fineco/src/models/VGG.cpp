#ifndef VGG_H_
#define VGG_H_

#include "models/model.h"
#include "kernel/layer.h"
#include "../backend/cuda/cuda_timer.cc"
#include "kernel/conv.h"
#include "../kernels/maxpool.cpp"
#include "../kernels/dense.cpp"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

class VGG final : public Model {
 public:
    
    VGG(const string& model_name = "VGG") : Model(model_name) {
        _sub_model_name = "VGG";
        if(model_name != "VGG16" && model_name != "VGG19" ) {
            cout<<"Model Init Failed, Model Name Must Be in {VGG16, VGG19}" <<endl;
        } 
    }
    
    void AddConv(int CO, int CI, int KH, int KW, int OH, int OW,  \
        const string& kernel_name, int N, int H, int W, bool need_input) {
        Conv2D* conv2d = new Conv2D(N, CI, H, W, CO, KH, KW, OH, OW, _runtime);
        // whether need input data
        if (need_input)
            conv2d->InitParams(true);
        else 
            conv2d->InitParams(false);
        ConfigureLayerCandidateByName(conv2d, _sub_model_name, kernel_name);
    }

    void AddMaxPool(int N, int C, int OH, int OW) {
        MaxPool* maxpool = new MaxPool(N, C, OH, OW);
        maxpool->InitParams();
        ConfigureLayerCandidateByName(maxpool, _sub_model_name, "maxpool_" + to_string(_maxpool_idx));
    }

    void AddDense(int In, int Out) {
        Dense* dense = new Dense(In, Out);
        dense->InitParams();
        ConfigureLayerCandidateByName(dense, _sub_model_name, "dense_" + to_string(_dense_idx));
    }

    // for block with 2 conv
    void AddBlockLayer1(int N, int CO, int CI, int OH, int OW) {
        // default conv1 changes output channel
        string kernel_name = "conv2d_block" + to_string(_conv2d_block_idx);
        if(_conv2d_block_idx == 1)
            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, true);
        else
            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);
        CI = CO;
        AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_2", N, OH, OW, false);
    }

    // for block with 3 conv and more
    void AddBlockLayer2(int N, int CO, int CI, int OH, int OW) {
        if (_conv2d_block_idx != 5) {
            string kernel_name = "conv2d_block" + to_string(_conv2d_block_idx);

            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);
            CI = CO;
            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_2", N, OH, OW, false);
            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_2", N, OH, OW, false);
            if (_model_name == "VGG19")
                AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_2", N, OH, OW, false);
        } else {
            string kernel_name = "conv2d_block" + to_string(_conv2d_block_idx);

            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);
            CI = CO;
            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);
            AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);
            if (_model_name == "VGG19")
                AddConv(CO, CI, 3, 3, OH, OW, kernel_name+"_1", N, OH, OW, false);
        }
    }


    void InitModel(RUNTIME runtime) {
        
        // Alloc Memory
        _runtime = runtime;

        // Add Block 1
        int N = 1, H = 224, W = 224, CO = 64, CI = 3, KH = 7, KW = 7, OH = 224, OW = 224;
        SetInPutSize(N*CI*H*W);
        AddBlockLayer1(N, CO, CI, OH, OW);
        _conv2d_block_idx++;
        
        CI = CO, OH = 112, OW = 112, H = 112, W = 112;
        AddMaxPool(N, CI, OH, OW);
        _maxpool_idx++;
  
    
        // Add Block 2
        CI = CO, CO = 128, KH = 3, KW = 3, OH = 112, OW = 112;
        AddBlockLayer1(N, CO, CI, OH, OW);
        _conv2d_block_idx++;

        CI = CO, OH = 56, OW = 56, H = 56, W = 56;
        AddMaxPool(N, CI, OH, OW);
       _maxpool_idx++;


        // Add Block 3
        CI = CO, CO = 256, KH = 3, KW = 3, OH = 56, OW = 56;
        AddBlockLayer2(N, CO, CI, OH, OW);
        _conv2d_block_idx++;

        CI = CO, OH = 28, OW = 28, H = 28, W = 28;
        AddMaxPool(N, CI, OH, OW);
        _maxpool_idx++;


        // Add Block 4
        CI = CO, CO = 512, KH = 3, KW = 3, OH = 28, OW = 28;
        AddBlockLayer2(N, CO, CI, OH, OW);
        _conv2d_block_idx++;

        CI = CO, OH = 14, OW = 14, H = 14, W = 14;
        AddMaxPool(N, CI, OH, OW);
        _maxpool_idx++;

        // Add Block 5
        CI = CO, CO = 512, KH = 3, KW = 3, OH = 14, OW = 14;
        AddBlockLayer2(N, CO, CI, OH, OW);
        _conv2d_block_idx++;

        CI = CO, OH = 7, OW = 7, H = 7, W = 7;
        AddMaxPool(N, CI, OH, OW);
        _maxpool_idx++;
        int Out = N * CO * OH * OW;
        SetOutPutSize(Out);

        cout<< _model_name + " Model Init Successfully"<<endl;
    }

 public:
  string _sub_model_name;
  int _conv2d_block_idx = 1;
  int _maxpool_idx = 1;
  int _dense_idx = 1;

};

#endif