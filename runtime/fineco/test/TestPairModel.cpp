
#include "../src/models/VGG.cpp"
#include "../src/models/AlexNet.cpp"
#include "../src/models/ResNet.cpp"
#include "test/test.h"

#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <thread>
#include <memory>
#include <future>
#include <chrono>   
using namespace std;
using namespace chrono;

class TestPairModel : public Test {
 public:

    TestPairModel(const string& task_1_name, const string& task_2_name) : _task_1_name(task_1_name), _task_2_name(task_2_name) {}

    template<typename M1, typename M2>
    void ServeBench(int sampling_num, int stream_num_1, int stream_num_2) {

        //AlexNet* _model_1 = new AlexNet(_task_1_name);
        //VGG* _model_2 = new VGG(_task_2_name);
        
        shared_ptr<M1> _model_1(new M1(_task_1_name));
        shared_ptr<M2> _model_2(new M2(_task_2_name));

        _model_1->InitModel(CUDA);
        _model_2->InitModel(CUDA);

        int request_num_1 = 100;
        int request_num_2 = 100;

        vector<int> task_1_rs;
        GenRequestStream(request_num_1, task_1_rs);
        vector<int> task_2_rs;
        GenRequestStream(request_num_2, task_2_rs);

        vector<int> task_1_rl;
        GenRandomList(GetRequestSize(task_1_rs), task_1_rl);
        vector<int> task_2_rl;
        GenRandomList(GetRequestSize(task_2_rs), task_2_rl);

        vector<int> task_1_candidate_cnt;
        vector<int> task_2_candidate_cnt;
        _model_1->GetAllLayerCandidateCnt(task_1_candidate_cnt);
        _model_2->GetAllLayerCandidateCnt(task_2_candidate_cnt);

        int all_request_num = GetRequestSize(task_1_rs) + GetRequestSize(task_2_rs);

        //warm up
        for (int i=0; i<10; i++) {
            vector<int> task_1_random_comb;
            vector<int> task_2_random_comb;

            GenDefaultCombination(task_1_candidate_cnt, task_1_random_comb);
            GenDefaultCombination(task_2_candidate_cnt, task_2_random_comb);
            
            // TODO dynamic change stream number
            int task_1_stream_num = 1;
            int task_2_stream_num = 1;

            vector<float> dnn_1_run_ret;
            vector<float> dnn_2_run_ret;


            future<float> task_1_latency = async(&Model::Run, _model_1, std::ref(task_1_random_comb), std::ref(task_1_rs),\
            _model_1->GetModelName(), task_1_stream_num, std::ref(task_1_rl), std::ref(dnn_1_run_ret),false);
            
            future<float> task_2_latency = async(&Model::Run, _model_2, std::ref(task_2_random_comb), std::ref(task_2_rs),\
            _model_2->GetModelName(), task_2_stream_num, std::ref(task_2_rl), std::ref(dnn_2_run_ret),false);

            float task_1_ret = task_1_latency.get();
            float task_2_ret = task_2_latency.get();
        }

        // balance request number
        for (int i=0; i<10; i++) {
            vector<int> task_1_random_comb;
            vector<int> task_2_random_comb;

            GenDefaultCombination(task_1_candidate_cnt, task_1_random_comb);
            GenDefaultCombination(task_2_candidate_cnt, task_2_random_comb);
            
            // TODO dynamic change stream number
            int task_1_stream_num = 1;
            int task_2_stream_num = 1;

            vector<float> dnn_1_run_ret;
            vector<float> dnn_2_run_ret;


            future<float> task_1_latency = async(&Model::Run, _model_1, std::ref(task_1_random_comb), std::ref(task_1_rs),\
            _model_1->GetModelName(), task_1_stream_num, std::ref(task_1_rl), std::ref(dnn_1_run_ret),false);
            
            future<float> task_2_latency = async(&Model::Run, _model_2, std::ref(task_2_random_comb), std::ref(task_2_rs),\
            _model_2->GetModelName(), task_2_stream_num, std::ref(task_2_rl), std::ref(dnn_2_run_ret),false);

            float lat_1 = task_1_latency.get();
            float lat_2 = task_2_latency.get();

            if(lat_1 > lat_2) {
                request_num_1 = 100;
                request_num_2 = 100 * (lat_1/lat_2);
            } else {
                request_num_2 = 100;
                request_num_1 = 100 * (lat_2/lat_1);
            }
        }
        
        request_num_1 *= stream_num_1;
        request_num_2 *= stream_num_2;
        cout<<"r1: "<<request_num_1<<" r2: "<<request_num_2<<endl;
        // re-generate request for balance 
        GenRequestStream(request_num_1, task_1_rs);
        GenRequestStream(request_num_2, task_2_rs);
        GenRandomList(GetRequestSize(task_1_rs), task_1_rl);
        GenRandomList(GetRequestSize(task_2_rs), task_2_rl);

        int repeat_num = 1;
        for (int i=0; i<sampling_num; i++) {
            if ( i%30 == 0 ) {
                cout<<"==== Current Running Samping ID : "<<i<<"===="<<endl;
            }
            for(int repeat=0; repeat<repeat_num; repeat++) {
                //auto start = system_clock::now();
                vector<int> task_1_random_comb;
                vector<int> task_2_random_comb;
                if(sampling_num > 1 && i==0) {
                    GenDefaultCombination(task_1_candidate_cnt, task_1_random_comb);
                    GenDefaultCombination(task_2_candidate_cnt, task_2_random_comb);
                } else {
                    //GenRandomCombination(task_1_candidate_cnt, task_1_random_comb);
                    //GenRandomCombination(task_2_candidate_cnt, task_2_random_comb);
                    //GenBestEffortCombination(task_1_candidate_cnt, task_1_random_comb);
                    //GenBestEffortCombination(task_2_candidate_cnt, task_2_random_comb);
                    GenDefaultCombination(task_1_candidate_cnt, task_1_random_comb);
                    GenDefaultCombination(task_2_candidate_cnt, task_2_random_comb);
                }

                int task_1_stream_num = stream_num_1;
                int task_2_stream_num = stream_num_2;

                vector<float> dnn_1_run_ret;
                vector<float> dnn_2_run_ret;

                /*
                task_1_random_comb[7] = 1;
                task_1_random_comb[8] = 1;
                task_1_random_comb[9] = 1;
                task_1_random_comb[12] = 2;
                task_1_random_comb[14] = 0;
                task_1_random_comb[15] = 1;
                task_1_random_comb[16] = 1;
                task_1_random_comb[17] = 1;
                task_1_random_comb[18] = 0;

                task_2_random_comb[7] = 1;
                task_2_random_comb[8] = 1;
                task_2_random_comb[9] = 1;
                task_2_random_comb[12] = 0;
                task_2_random_comb[14] = 0;
                task_2_random_comb[15] = 1;
                task_2_random_comb[16] = 1;
                task_2_random_comb[17] = 2;
                task_2_random_comb[18] = 0;
                */


                future<float> task_1_latency = async(&Model::Run, _model_1, std::ref(task_1_random_comb), std::ref(task_1_rs),\
                _model_1->GetModelName(), task_1_stream_num, std::ref(task_1_rl), std::ref(dnn_1_run_ret), false);
            
                future<float> task_2_latency = async(&Model::Run, _model_2, std::ref(task_2_random_comb), std::ref(task_2_rs),\
                _model_2->GetModelName(), task_2_stream_num, std::ref(task_2_rl),std::ref(dnn_2_run_ret), false);

                float task_1_ret = task_1_latency.get();
                float task_2_ret = task_2_latency.get();

                task_1_ret_list[i].push_back(task_1_ret);
                task_2_ret_list[i].push_back(task_2_ret);
                //cout<<"alexnet: "<<alex_ret<<" "<<"resnet: "<<resnet_ret<<endl;

                //auto end   = system_clock::now();
                //auto duration = duration_cast<microseconds>(end - start);
                //cout << i<< "cost " << double(duration.count()) * milliseconds::period::num  << "ms" << endl;
                //float cost_s = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                float throughput = dnn_1_run_ret[0] + dnn_2_run_ret[0];
                co_tp_list[i].push_back(throughput);
            }
           
        }

        

        for(auto& item : task_1_ret_list) {
            float avg_lat = 0.0;
            for (auto lat : item.second) {
                avg_lat += lat;
            }
            avg_lat /= repeat_num;
            //cout<<"AlexNet "<<item.first<<" avg latency:"<<avg_lat;
            task_1_avg_lat_list.push_back(avg_lat);
        }

        for(auto& item : task_2_ret_list) {
            float avg_lat = 0.0;
            for (auto lat : item.second) {
                avg_lat += lat;
            }
            avg_lat /= repeat_num;
            //cout<<"Resnet "<<item.first<<" avg latency:"<<avg_lat;
            task_2_avg_lat_list.push_back(avg_lat);
        }

        float default_tp = 0.0;
        float default_task_1_lat = task_1_avg_lat_list[0];
        float default_task_2_lat = task_2_avg_lat_list[0];

        float best_tp = 0.0;
        int best_tp_id = -1;

        // Get Max Throughput
        for(auto& item : co_tp_list) {
            float avg_tp = 0.0;
            for (auto tp : item.second) {
                avg_tp += tp;
            }
            avg_tp /= repeat_num;
            throughput_list.push_back(avg_tp);
            int comb_id = item.first;
            float lat_1 = task_1_avg_lat_list[comb_id];
            float lat_2 = task_2_avg_lat_list[comb_id];
            if (avg_tp > best_tp) {
                best_tp = avg_tp;
                best_tp_id = comb_id;
            }

            if(item.first == 0) default_tp = avg_tp;
        }

        // Get Min Latency
        float min_latency = 999;
        int min_lat_idx = -1;
        for(int i=0; i<task_1_avg_lat_list.size(); i++) {
            float lat1 = task_1_avg_lat_list[i];
            float lat2 = task_2_avg_lat_list[i];
            if (min_latency > lat1 + lat2) {
                min_lat_idx = i;
                min_latency = lat1 + lat2;
            }
        }


        cout<<"Best Tp is: "<<best_tp<<" "<<_model_1->GetModelName()<<"avg latence:"<<task_1_avg_lat_list[best_tp_id]<<" "<<_model_2->GetModelName()<<" avg latence:"<<task_2_avg_lat_list[best_tp_id]<<endl;
        cout<<"Best Qos Coresponse Tp is: "<<throughput_list[min_lat_idx]<<" Best Latency 1: "<<task_1_avg_lat_list[min_lat_idx]<<" Best Latency 2: "<<task_2_avg_lat_list[min_lat_idx]<<endl;
        cout<<"Default Tp is: "<<default_tp<<" "<<_model_1->GetModelName()<<" avg latence:"<<default_task_1_lat<< " "<<_model_2->GetModelName()<<" avg latence:"<<default_task_2_lat<<endl;
        //delete _model_1;
        //delete _model_2;
    }

    private:
        string _task_1_name;
        string _task_2_name;

        unordered_map<int, vector<float>> task_1_ret_list;
        unordered_map<int, vector<float>> task_2_ret_list;
        unordered_map<int, vector<float>> co_tp_list;

        vector<float> task_1_avg_lat_list;
        vector<float> task_2_avg_lat_list;
        vector<float> throughput_list;
    

};