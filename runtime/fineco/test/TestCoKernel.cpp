#include <chrono>
#include<algorithm>

class TestCoKernel : public Test {
 public:

    TestCoKernel(const string& task_1_name, const string& task_2_name) : _task_1_name(task_1_name),\
     _task_2_name(task_2_name) {}
    
     template<typename M1, typename M2>
     void KernelBench() {
        
        shared_ptr<M1> _model_1(new M1(_task_1_name));
        shared_ptr<M2> _model_2(new M2(_task_2_name));

        _model_1->InitModel(CUDA);
        _model_2->InitModel(CUDA);

        vector<int> task_1_candidate_cnt;
        vector<int> task_2_candidate_cnt;
        _model_1->GetAllLayerCandidateCnt(task_1_candidate_cnt);
        _model_2->GetAllLayerCandidateCnt(task_2_candidate_cnt);

        ofstream fout;
        fout.open("stats_"+ _task_1_name + "_" + _task_2_name + ".log");

        for(int model_1_layer=0; model_1_layer<task_1_candidate_cnt.size(); model_1_layer++) {
            for(int model_2_layer=0; model_2_layer<task_2_candidate_cnt.size(); model_2_layer++){
                if(_model_1->GetLayerName(model_1_layer) == "MAXPOOL" || _model_2->GetLayerName(model_2_layer)== "MAXPOOL") continue;
                //if(task_1_candidate_cnt[model_1_layer] == 1 && task_2_candidate_cnt[model_2_layer] == 1) continue;
                if (model_1_layer != model_2_layer) continue;
                cout<<"====================="<<endl;
                cout<<_model_1->GetModelName()<<" layer: "<<model_1_layer<<endl;
                cout<<_model_2->GetModelName()<<" layer: "<<model_2_layer<<endl;
                cout<<"====================="<<endl;
                fout<<"layer pair( "<<model_1_layer<<", "<<model_2_layer<<" )"<<endl;

                for(int model_1_layer_candidate=0; model_1_layer_candidate<task_1_candidate_cnt[model_1_layer]; model_1_layer_candidate++){
                    for(int model_2_layer_candidate=0; model_2_layer_candidate<task_2_candidate_cnt[model_2_layer]; model_2_layer_candidate++) {
                        if(model_1_layer_candidate != model_2_layer_candidate) continue;
                        // warm up
                        auto a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate, nullptr);
                        auto a2 = async(&Model::LayerRunWarmup, _model_2, model_2_layer, model_2_layer_candidate, nullptr);
                        float lat_1 = a1.get();
                        float lat_2 = a2.get();

                        // Single Run
                        a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate, nullptr);
                        float single_lat_1 = a1.get();

                        a2 = async(&Model::LayerRunWarmup, _model_2, model_2_layer, model_2_layer_candidate, nullptr);
                        float single_lat_2 = a2.get();
                        
                        cout<<"Single Run candidate: "<<model_1_layer_candidate<<" Avg Latency: "<<single_lat_1/500<<\
                        " candidate: "<<model_2_layer_candidate<<" Avg Latency: "<<single_lat_2/500<<endl;

                        fout<<"    Candidate Pair( "<<model_1_layer_candidate<<", "<<model_2_layer_candidate<<" )"<<endl;
                        fout<<"    Isolation Latency : "<<single_lat_1/500<<", "<<single_lat_2/500<<endl;

                        // Get Co-locate Task Number
                        a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate, nullptr);
                        a2 = async(&Model::LayerRunWarmup, _model_2, model_2_layer, model_2_layer_candidate, nullptr);
                        lat_1 = a1.get();
                        lat_2 = a2.get();

                        cout<<"Co Run without full overlap: "<<model_1_layer_candidate<<" Avg Latency: "<<lat_1/500<<\
                        " candidate: "<<model_2_layer_candidate<<" Avg Latency: "<<lat_2/500<<endl;


                        int request_num_1 = 1;
                        int request_num_2 = 1;
                        int base_num = 1000;
                        if(lat_1 > lat_2) {
                            request_num_1 = base_num;
                            request_num_2 = base_num * (lat_1/lat_2);
                        } else {
                            request_num_2 = 100;
                            request_num_1 = 100 * (lat_2/lat_1);
                        }
                        
                        auto start = chrono::steady_clock::now();
                        float task_1_ret = 0;
                        float task_2_ret = 0;
                        
                        future<float> task_1_latency = async(&Model::LayerRun, _model_1, model_1_layer, model_1_layer_candidate, request_num_1, nullptr);
                        future<float> task_2_latency = async(&Model::LayerRun, _model_2, model_2_layer, model_2_layer_candidate, request_num_2, nullptr);
                        task_1_ret += task_1_latency.get();
                        task_2_ret += task_2_latency.get();


                        cout<<"candidate: "<<model_1_layer_candidate<<" Total Latency: "<<task_1_ret<<" Average Latency: "<<task_1_ret/request_num_1\
                        <<" candidate: "<<model_2_layer_candidate<<" Total Latency: "<<task_2_ret<<" Average Latency: "<<task_2_ret/request_num_2<<endl;

                        fout<<"    CO Run Avg Latency: "<<task_1_ret/request_num_1<<", "<<task_2_ret/request_num_2<<endl;

                        auto end = chrono::steady_clock::now();
                        float co_latency = chrono::duration_cast<chrono::microseconds>(end - start).count();
                        cout << "Co Running time: "<< co_latency << " us" << endl;

                        fout<<"    Co Run Total Time: "<< co_latency<<endl;
                    }
                }       
            }
        }
        fout.close();
        cout<<"Kernel Co Bench End"<<endl;
    }

    template<typename M1, typename M2>
    void KernelTest(int model_1_layer, int model_2_layer, int model_1_layer_candidate, int model_2_layer_candidate) {
        
        shared_ptr<M1> _model_1(new M1(_task_1_name));
        shared_ptr<M2> _model_2(new M2(_task_2_name));

        _model_1->InitModel(CUDA);
        _model_2->InitModel(CUDA);

        // warm up
        auto a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate, nullptr);
        auto a2 = async(&Model::LayerRunWarmup, _model_2, model_2_layer, model_2_layer_candidate, nullptr);
        float lat_1 = a1.get();
        float lat_2 = a2.get();
        
        float min_task_1_ret = 999;
        float min_task_2_ret = 999;
        vector<float> task_1_ret_list;
        vector<float> task_2_ret_list;
        float solo_ret_1 = 0;
        float solo_ret_2 = 0;
        auto stream1 =  GetBackendHandle(CUDA)->CreateStream(0);
        auto stream2 =  GetBackendHandle(CUDA)->CreateStream(0);
        int repeat_num = 100;
        int iter_num = 100;
        int warm_up_num = 50;
        for (int i=0; i<iter_num; i++) {
            future<float> task_1_latency = async(&Model::LayerRun, _model_1, model_1_layer, model_1_layer_candidate, repeat_num, stream1);
            future<float> task_2_latency = async(&Model::LayerRun, _model_2, model_2_layer, model_2_layer_candidate, repeat_num, stream2);
            auto task_1_ret = task_1_latency.get();
            auto task_2_ret = task_2_latency.get();
            if (i < warm_up_num) continue;
            min_task_1_ret = min_task_1_ret > task_1_ret ? task_1_ret : min_task_1_ret;
            task_1_ret_list.push_back(task_1_ret);
            min_task_2_ret = min_task_2_ret > task_2_ret ? task_2_ret : min_task_2_ret;
            task_2_ret_list.push_back(task_2_ret);
            //cout<<"task_1_ret: "<<task_1_ret<<" task_2_ret: "<<task_2_ret<<endl;
        }

        future<float> task_1_latency = async(&Model::LayerRun, _model_1, model_1_layer, model_1_layer_candidate, repeat_num, stream1);
        solo_ret_1 = task_1_latency.get();

        future<float> task_2_latency = async(&Model::LayerRun, _model_2, model_2_layer, model_2_layer_candidate, repeat_num, stream2);
        solo_ret_2 = task_2_latency.get();

        solo_ret_1 = (solo_ret_1 * 1000) / repeat_num;
        solo_ret_2 = (solo_ret_2 * 1000) / repeat_num;
        
        cout<<"candidate: "<<model_1_layer_candidate<<" Solo Latency: "<<solo_ret_1<<\
        " candidate: "<<model_2_layer_candidate<<" Solo Latency: "<<solo_ret_2<<endl;

        //float tp_1 = ((repeat_num * iter_num) / task_1_ret) * 1000;
        //float tp_2 = ((repeat_num * iter_num)  / task_2_ret) * 1000;
        min_task_1_ret = (min_task_1_ret * 1000) / repeat_num;
        min_task_2_ret = (min_task_2_ret * 1000) / repeat_num;
        sort(task_1_ret_list.begin(), task_1_ret_list.end());
        sort(task_2_ret_list.begin(), task_2_ret_list.end());
        auto task_1_95_ret = task_1_ret_list[warm_up_num*0.9];
        auto task_2_95_ret = task_2_ret_list[warm_up_num*0.9];
        task_1_95_ret *= 1000 / repeat_num;
        task_2_95_ret *= 1000 / repeat_num;
        cout<<"candidate: "<<model_1_layer_candidate<<" min Co Latency us : "<<min_task_1_ret<<\ 
        " 95% Co Latency us : "<<task_1_95_ret<<endl<<\
        " candidate: "<<model_2_layer_candidate<<" min Co Latency us: "<<min_task_2_ret<<\
        " 95% Co Latency us : "<<task_2_95_ret<<endl;
    }

 private:
   string _task_1_name;
   string _task_2_name;


};