#ifndef RESNET_H_
#define RESNET_H_


#include "models/model.h"
#include "kernel/layer.h"
#include "../backend/rocm/rocm_timer.cc"
#include "kernel/conv.h"
#include "../kernels/maxpool.cpp"
#include "../kernels/dense.cpp"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

class ResNet final : public Model {
 public:
    
    ResNet(const string& model_name = "ResNet") : Model(model_name) {
        if(_model_name == "ResNet18" || _model_name == "ResNet34" ) {
            _sub_model_name = "ResNet18a34";
        } else if (_model_name == "ResNet50" || _model_name == "ResNet101" || _model_name == "ResNet152") {
            _sub_model_name = "ResNet50s";
        } else {
            cout<<"Model Init Failed, Model Name Must Be in {ResNet18, ResNet34, ResNet50, ResNet101, ResNet152}" <<endl;
        }

    }
    
    void AddConv(int CO, int CI, int KH, int KW, int OH, int OW,  \
        const string& kernel_name, int N, int H, int W, bool need_input) {
        Conv2D* conv2d = new Conv2D(N, CI, H, W, CO, KH, KW, OH, OW, _runtime);
        conv2d->InitParams(need_input);
        ConfigureLayerCandidateByPath(conv2d, _sub_model_name, kernel_name);
    }

    void AddMaxPool(int N, int C, int OH, int OW) {
        MaxPool* maxpool = new MaxPool(N, C, OH, OW);
        maxpool->InitParams();
        ConfigureLayerCandidateByPath(maxpool, _sub_model_name, "maxpool");
    }

    void AddDense(int In, int Out) {
        Dense* dense = new Dense(In, Out);
        dense->InitParams();
        ConfigureLayerCandidateByPath(dense, _sub_model_name, "dense");
    }

    void AddBasicBlockLayer(int N, int CO, int CI, int OH, int OW, int repeat) {
        for (int i=0; i<repeat; i++) {
            if (i == 0 && _layer_idx > 2) {
                // For resnet18 and resnet34 use conv1 to represent downsampling
                int H = OH * 2;
                int W = OW * 2;
                AddConv(CO, CI, 3, 3, OH, OW, "conv_ds", N, H, W, false);
                CI = CO;
            } else {
                AddConv(CO, CI, 3, 3, OH, OW, "conv1", N, OH, OW, false);
            }
            
            AddConv(CO, CI, 3, 3, OH, OW, "conv1", N, OH, OW, false);
        }
    }

    void AddBottleneckLayer(int N, int CO, int CI, int OH, int OW, int repeat) {
        int ori_co = CO;
        for (int i=0; i<repeat; i++) {
            CO = ori_co;
            if (i == 0)
                AddConv(CO, CI, 1, 1, OH, OW, "conv1", N, OH, OW, false);
            else  
                AddConv(CO, CI, 1, 1, OH, OW, "conv4", N, OH, OW, false);
            CI = CO;

            // For resnet50 ... use conv2_ds to represent downsampling
            if(_layer_idx > 2 && i == 0) {
                AddConv(CO, CI, 3, 3, OH/2, OW/2,  "conv2_ds", N, OH, OW, false);
                OH /= 2;
                OW /= 2;
            } else {
                AddConv(CO, CI, 3, 3, OH, OW, "conv2", N, OH, OW, false);
            }

            CI = CO;
            CO = CO * 4;
            AddConv(CO, CI, 1, 1, OH, OW, "conv3", N, OH, OW, false);
            CI = CO;
        }
    }

    void InitModel(RUNTIME runtime) {
        
        // Alloc Memory
        _runtime = runtime;
        if (_runtime == ROCM) {
            if (_model_name == "ResNet152" || _model_name == "ResNet50")
                SetRightSize(64);
        }

        // Begin Add Common Layer
        int N = 1, H = 224, W = 224, CO = 64, CI = 3, KH = 7, KW = 7, OH = 112, OW = 112;
        SetInPutSize(N*CI*H*W);
        AddConv(CO, CI, KH, KW, OH, OW, "conv1", N, H, W, true);
        _layer_idx++;
        
        CI = CO, OH = 56, OW = 56, H = 56, W = 56;
        AddMaxPool(N, CI, OH, OW);
        // End Add Common Layer

        CI = CO, CO = 64, KH = 3, KW = 3, OH = 56, OW = 56;

        vector<int> block_serise;
        if (_model_name == "ResNet18" || _model_name == "ResNet34" ) {
            if (_model_name == "ResNet18")
                block_serise = {2,2,2,2};
            else
                block_serise = {3,4,6,3};

            for(int i=0; i<block_serise.size(); i++) {
                // layer2 (equals i=0) don't need change input shape and channel
                if(i != 0)
                    CI = CO, CO *= 2, OH /= 2, OW /= 2;
                AddBasicBlockLayer(N, CO, CI, OH, OW, block_serise[i]);
                _layer_idx++;
            }
        } else if (_model_name == "ResNet50" || _model_name == "ResNet101" || _model_name == "ResNet152") {
            if (_model_name == "ResNet50")
                block_serise = {3,4,6,3};
            else if (_model_name == "ResNet101")
                block_serise = {3,4,23,3};
            else
                block_serise = {3,8,36,3};
            vector<int> last_layer_dim = {0, 256, 512, 1024};
            for(int i=0; i<block_serise.size(); i++) {
                if(i != 0)
                    CI = last_layer_dim[i], CO *= 2;
                AddBottleneckLayer(N, CO, CI, OH, OW, block_serise[i]);
                if (i != 0)
                    OH /= 2, OW /= 2;
                _layer_idx++;
            }
        }

        
        //int In = N * CO * OW * OW;
        //int Out = 1000;
        //AddDense(In, Out);
        //_layer_idx++;

        int Out = N * CO * OH * OW;
        SetOutPutSize(Out);

        // cout<< _model_name + " Model Init Successfully"<<endl;
    }

 public:
  string _sub_model_name;

};

#endif
