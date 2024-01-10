#include "kernel/layer.h"
#include "../backend/cuda/cuda_timer.cc"
#include "../kernels/conv2d.cpp"
#include "../kernels/maxpool.cpp"
#include "../kernels/dense.cpp"
#include <iostream>

class LayerCoRunner {
 public:
    AddLayer(Layer* layer) {
        _run_list.push_back(layer);
    }

    InitParam() {
        for(auto layer : _run_list) {

        }
    }

    Run(int repeats) {
        auto candidate_name = layer->GetCandidateNameById(candidate_idx);
        auto layer_kind = layer->GetLayerKind();
        float* intermediate_data;
        if (layer_kind == CONV_RELU) {
            try {
                Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);
                conv2d->Compute(candidate_name, _stream);
                intermediate_data = conv2d->GetOutPut();
            } catch(std::bad_cast const& ex) {
                cout << "[ Conv2D"<<ex.what()<<"]" << endl;
            }    
        } else if (layer_kind == MAXPOOL) {
            try {
                MaxPool* maxpool = dynamic_cast<MaxPool*>(layer);
                maxpool->Compute(candidate_name, _stream, intermediate_data);
                intermediate_data = maxpool->GetOutPut();
            } catch(std::bad_cast const& ex) {
                cout << "[ Maxpool"<<ex.what()<<"]" << endl;
            } 
        }  else if (layer_kind == DENSE) {
            try {
                Dense* dense = dynamic_cast<Dense*>(layer);
                dense->Compute(candidate_name, _stream, intermediate_data);
                intermediate_data = dense->GetOutPut();
            } catch(std::bad_cast const& ex) {
                cout << "[ Maxpool"<<ex.what()<<"]" << endl;
            } 
        }  
    }



 private:
    vector<Layer*> _run_list;



};
