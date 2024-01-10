#include <chrono>


class TestOneKernel : public Test {
 public:
    
     template<typename M1>
     void KernelBench(const string& model_name) {
        
        shared_ptr<M1> _model_1(new M1(model_name));

        _model_1->InitModel(ROCM);

        vector<int> task_1_candidate_cnt;
        _model_1->GetAllLayerCandidateCnt(task_1_candidate_cnt);

        // ofstream fout;
        // fout.open("stats_"+ model_name + ".log");

        for(int model_1_layer=0; model_1_layer<task_1_candidate_cnt.size(); model_1_layer++) {
            if(_model_1->GetLayerName(model_1_layer) == "MAXPOOL") continue;
            //if(task_1_candidate_cnt[model_1_layer] == 1 && task_2_candidate_cnt[model_2_layer] == 1) continue;
            // fout<<"layer ["<<model_1_layer<<"]"<<endl;

            int model_1_layer_candidate = task_1_candidate_cnt[model_1_layer] - 1;

            // warm up
            auto a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate, nullptr);
            float lat_1 = a1.get();

            // Single Run
            a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate, nullptr);
            float single_lat_1 = a1.get();

            cout<<_model_1->GetModelName()<<" layer: "<<model_1_layer<<\
                ", Avg Latency: "<<single_lat_1/100<<endl;

            // fout<<"    Isolation Latency : "<<single_lat_1/100<<endl;
    
        }
        // fout.close();
    }

    // template<typename M1>
    // void KernelTest(int model_1_layer, int model_1_layer_candidate) {
        
    //     shared_ptr<M1> _model_1(new M1(_task_1_name));

    //     _model_1->InitModel(ROCM);

    //     // warm up
    //     auto a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate);
    //     float lat_1 = a1.get();
        
    //     auto start = chrono::steady_clock::now();
    //     float task_1_ret = 0;
    //     float solo_ret_1 = 0;
    //     auto stream1 =  GetBackendHandle(ROCM)->CreateStream(0);
    //     for (int i=0; i<100; i++) {
    //         future<float> task_1_latency = async(&Model::LayerRun, _model_1, model_1_layer, model_1_layer_candidate, 5, stream1);
    //         task_1_ret += task_1_latency.get();
    //     }

    //     future<float> task_1_latency = async(&Model::LayerRun, _model_1, model_1_layer, model_1_layer_candidate, 5, stream1);
    //     solo_ret_1 = task_1_latency.get();
        
    //     cout<<"candidate: "<<model_1_layer_candidate<<" Latency: "<<solo_ret_1<<endl;

    //     float tp_1 = (100 / task_1_ret) * 1000;
    //     cout<<"candidate: "<<model_1_layer_candidate<<" Latency: "<<task_1_ret<< " TP: "<<tp_1<<endl;

    //     auto end = chrono::steady_clock::now();
    //     auto co_latency = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000;
    //     cout << "Co Running time: "<< co_latency << " ms" << endl;

    // }

 private:


};