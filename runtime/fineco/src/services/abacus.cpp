#include "service/service.h"
#include "common/thread_pool.hpp"
#include "common/gpu_stats.hpp"
#include <algorithm>

class AbacusService : public PairService {
 public: 

    void Accept() {
        cout<<"==== Start Abacus Service ========"<<endl;

        vector<int> candidate_cnt_1;
        _model_1->GetAllLayerCandidateCnt(candidate_cnt_1);
        vector<int> candidate_high_selector_1;
        ServiceGenDefaultCombination(candidate_cnt_1, candidate_high_selector_1);
        //ServiceGenCustomCombination(candidate_cnt_1, candidate_high_selector_1);
        vector<int> candidate_low_selector_1;
        ServiceGenDefaultCombination(candidate_cnt_1, candidate_low_selector_1);
        //ServiceGenCustomCombination(candidate_cnt_1, candidate_low_selector_1);

         vector<int> candidate_cnt_2;
        _model_2->GetAllLayerCandidateCnt(candidate_cnt_2);
        vector<int> candidate_high_selector_2;
        ServiceGenDefaultCombination(candidate_cnt_2, candidate_high_selector_2);
        //ServiceGenCustomCombination(candidate_cnt_2, candidate_high_selector_2);
        vector<int> candidate_low_selector_2;
        ServiceGenDefaultCombination(candidate_cnt_2, candidate_low_selector_2);
        //ServiceGenCustomCombination(candidate_cnt_2, candidate_low_selector_2);

        int default_stream_num = 10;
        int high_priority_stream_num = 0;
        int pool_size = default_stream_num + high_priority_stream_num;
        ThreadPool client_pool(pool_size);

        
        int stream_num = pool_size;
        vector<float*> _output_memory_pool_1(stream_num);
        vector<float*> _input_memory_pool_1(stream_num);
        vector<float*> _output_memory_pool_2(stream_num);
        vector<float*> _input_memory_pool_2(stream_num);

        vector<DLStream> _stream_pool(default_stream_num);
        vector<DLStream> _high_stream_pool(high_priority_stream_num);


        size_t out_mem_size_1 = sizeof(float) * _model_1->GetOutputSize();
        size_t input_mem_size_1 = sizeof(float) * _model_1->GetInputSize();
        size_t out_mem_size_2 = sizeof(float) * _model_2->GetOutputSize();
        size_t input_mem_size_2 = sizeof(float) * _model_2->GetInputSize();


        for(int i=0; i<stream_num; i++) {
            float* out_h_1 = (float*)malloc(out_mem_size_1);
            _output_memory_pool_1[i] = out_h_1;
            float* in_h_1 = (float*)malloc(input_mem_size_2);
            RandomInit(in_h_1, _model_1->GetInputSize());
            _input_memory_pool_1[i] = in_h_1;

            float* out_h_2 = (float*)malloc(out_mem_size_2);
            _output_memory_pool_2[i] = out_h_2;
            float* in_h_2 = (float*)malloc(input_mem_size_2);
            RandomInit(in_h_2, _model_2->GetInputSize());
            _input_memory_pool_2[i] = in_h_2;
            if(i < default_stream_num)
                _stream_pool[i] = GetBackendHandle(_model_1->GetModelRuntime())->CreateStream(0);
        }

        for(int i=0; i<high_priority_stream_num; i++) {
            _high_stream_pool[i] = GetBackendHandle(_model_1->GetModelRuntime())->CreateStreamWithPriority(-5, 0);
        }
        
        
        
        int model_1_number = 0;
        int model_2_number = 0;

        // flexible sycn client threads
        int acc_model1_number = 0;
        

        float model_1_latency = 0.0 ;
        float model_2_latency = 0.0;
        float high_qos_latency = 0.0;
        float high_qos_latency_no_wait = 0.0;
        int high_qos_cnt = 0;

        vector<float> model_1_lat_list;
        vector<float> model_2_lat_list;

        for(int i=0; i<20; i++) {
            auto results_1 = _model_1->RealRun(candidate_high_selector_1, _output_memory_pool_1[0],\
                    _input_memory_pool_1[0], input_mem_size_1, _stream_pool[0],true);
            if(i>=10) model_1_latency += results_1;
        }

        for(int i=0; i<20; i++) {
            auto results_2 = _model_2->RealRun(candidate_high_selector_2, _output_memory_pool_2[0],\
                    _input_memory_pool_2[0], input_mem_size_2, _stream_pool[0], true);
            if(i>=10) model_2_latency += results_2;
        }

        model_1_latency /= 10;
        model_2_latency /= 10;
        cout<<"model 1 isolation latency: "<<model_1_latency<<" model 2 isolation latency :"<<model_2_latency<<endl;
        
        int64_t first_submit_timestamp = 0;
        int arrival_interval_us = 20000;
        float avg_wait_time = 0;
        vector<float> wait_time_1;
        vector<float> wait_time_2;
        float peak_tp = 0;
        int peak_reqs = 0;

        int high_qos_cnt_1 = 0;
        int high_qos_cnt_2 = 0;

        vector<int64_t> co_stream_list;

        auto outer_start = system_clock::now();

        //auto start = system_clock::now();
        while(_total_request_queue.size() > 0) {
            // concurrent submit request by stream number
            //cout<<"left request size: "<<_total_request_queue.size()<<endl;
            int submit_number = 0;
            int first_timestamp = _total_request_queue[0]->GetGeneTimestep();
            int stream_id = 0;
            vector<future<float>> results_model1;
            vector<future<float>> results_model2;
            vector<shared_ptr<Request>> tmp_req;
            for(int ri=0; ri<_total_request_queue.size(); ri++) {
                if(_total_request_queue[ri]->GetGeneTimestep() == first_timestamp) {
                    tmp_req.push_back(_total_request_queue[ri]);
                }
            }
            
            //cout<<"tmp req size: "<<tmp_req.size()<<endl;
            auto inner_start = system_clock::now();

            for(auto cur_request : tmp_req) {
                if (stream_id >= default_stream_num) stream_id = stream_id % default_stream_num;
                auto stream = _stream_pool[stream_id];
                auto submit_time = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count();
                if (first_submit_timestamp == 0) first_submit_timestamp = submit_time;
                auto arrivel_time = cur_request->GetGeneTimestep()*arrival_interval_us + first_submit_timestamp;
                if (cur_request->GetModelKind() == M1 && submit_time > arrivel_time && cur_request->GetQoSKind() != Empty) {
                    float wait_time = (submit_time - arrivel_time)/1000;
                    //cout<<"wait time: "<<wait_time<<endl;
                    wait_time_1.emplace_back(wait_time);
                    submit_number++;
                    if (cur_request->GetQoSKind() == HighQoS) {
                        results_model1.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_high_selector_1), _output_memory_pool_1[stream_id],\
                        _input_memory_pool_1[stream_id], input_mem_size_1, stream, true));
                        high_qos_cnt_1++;
                    } else {
                        results_model1.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_low_selector_1), _output_memory_pool_1[stream_id],\
                        _input_memory_pool_1[stream_id], input_mem_size_1, stream, true));
                    }
                    model_1_number++;
                    stream_id++;
                    _total_request_queue.pop_front();
                    //cout<<"request size: "<<_total_request_queue.size()<<endl;
                } else if (cur_request->GetModelKind() == M2 && submit_time > arrivel_time && cur_request->GetQoSKind() != Empty) {
                    float wait_time = (submit_time - arrivel_time)/1000;
                    wait_time_2.emplace_back(wait_time);
                    submit_number++;
                    if (cur_request->GetQoSKind() == HighQoS) {
                        results_model2.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_high_selector_2), _output_memory_pool_2[stream_id],\
                        _input_memory_pool_2[stream_id], input_mem_size_2, stream, true));
                        high_qos_cnt_2++;
                    } else {
                        results_model2.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_low_selector_2), _output_memory_pool_2[stream_id],\
                        _input_memory_pool_2[stream_id], input_mem_size_2, stream, true));
                    }
                    model_2_number++;
                    stream_id++;
                    _total_request_queue.pop_front();
                }
                
                
            }

            for(int i=0; i<results_model1.size(); i++) {
                auto r = results_model1[i].get();
                model_1_lat_list.push_back(r);
            }

            for(int i=0; i<results_model2.size(); i++) {
                auto r = results_model2[i].get();
                model_2_lat_list.push_back(r);
            }


            auto inner_end = system_clock::now();
            auto duration = duration_cast<microseconds>(inner_end - inner_start).count();
            
            if(duration > 1000){
                co_stream_list.push_back(duration);
                //cout<<"process request number: "<<tmp_req.size()<<" duration: "<<duration<<" "<<endl;
                float tmp_tp = (tmp_req.size() * 1000 * 1000 ) / duration;
                if (peak_reqs <= tmp_req.size()) {
                    peak_reqs = tmp_req.size();
                    peak_tp = max(tmp_tp, peak_tp);
                }             
            }
        }

        auto outer_end = system_clock::now();
        auto duration = duration_cast<milliseconds>(outer_end - outer_start).count();
        float throughput = (model_1_number + model_2_number) * 1000 / duration;
        //float throughput = (total_cnt * 1000) / model_1_avg_lat;
        cout << "run cost " << duration << "ms" << " throughput:" << throughput << "request/s" <<endl;
        
        

        float model_1_avg_lat = 0.0;
        float model_2_avg_lat = 0.0;
        float model_1_avg_lat_no_wait = 0.0;
        float model_2_avg_lat_no_wait = 0.0;
        float avg_wait_time_1 = 0;
        float avg_wait_time_2 = 0;
        int warm_up = 3;
        int total_cnt_1 = wait_time_1.size()-warm_up;
        int total_cnt_2 = wait_time_2.size()-warm_up;
        float total_latency_1 = 0;
        float total_latency_2 = 0;

        vector<float> model_1_wait_list;
        vector<float> model_2_wait_list;

        for(int i=warm_up; i<total_cnt_1+warm_up; i++) {
            float tmp_wait_time = wait_time_1[i];
            avg_wait_time_1 += tmp_wait_time;
            float latency = model_1_lat_list[i];
            auto latency_add_wait = latency + tmp_wait_time;
            model_1_avg_lat += latency_add_wait;
            model_1_wait_list.push_back(latency_add_wait);
            model_1_avg_lat_no_wait += latency;
            total_latency_1 += model_1_avg_lat;
        }
        model_1_avg_lat /= total_cnt_1;
        model_1_avg_lat_no_wait /= total_cnt_1;
        avg_wait_time_1 /= total_cnt_1;

        for(int i=warm_up; i<total_cnt_2+warm_up; i++) {
            float tmp_wait_time = wait_time_2[i];
            avg_wait_time_2 += tmp_wait_time;
            float latency = model_2_lat_list[i];
            auto latency_add_wait = latency + tmp_wait_time;
            model_2_avg_lat += latency_add_wait;
            model_2_wait_list.push_back(latency_add_wait);
            model_2_avg_lat_no_wait += latency;
            total_latency_2 += model_2_avg_lat;
        }
        model_2_avg_lat /= total_cnt_2;
        model_2_avg_lat_no_wait /= total_cnt_2;
        avg_wait_time_2 /= total_cnt_2;

        cout<<"avg wait time 1 : "<<avg_wait_time_1<<" avg wait time 2: "<<avg_wait_time_2<<endl;
        
        float model_1_low_whisker = 0;
        float model_1_low_quartile = 0;
        float model_1_high_whisker = 0;
        float model_1_high_quartile = 0;
        float model_1_median = 0;

        float model_2_low_whisker = 0;
        float model_2_low_quartile = 0;
        float model_2_high_whisker = 0;
        float model_2_high_quartile = 0;
        float model_2_median = 0;
        
        sort(model_1_lat_list.begin()+warm_up, model_1_lat_list.end());
        sort(model_2_lat_list.begin()+warm_up, model_2_lat_list.end());

        model_1_low_whisker = model_1_lat_list[warm_up+total_cnt_1*0.05];
        model_1_low_quartile = model_1_lat_list[warm_up+total_cnt_1*0.25];
        model_1_high_whisker = model_1_lat_list[warm_up+total_cnt_1*0.95];
        model_1_high_quartile = model_1_lat_list[warm_up+total_cnt_1*0.75];
        model_1_median = model_1_lat_list[warm_up+total_cnt_1*0.5];


        model_2_low_whisker = model_2_lat_list[warm_up+total_cnt_2*0.05];
        model_2_low_quartile = model_2_lat_list[warm_up+total_cnt_2*0.25];
        model_2_high_whisker = model_2_lat_list[warm_up+total_cnt_2*0.95];
        model_2_high_quartile = model_2_lat_list[warm_up+total_cnt_2*0.75];
        model_2_median = model_2_lat_list[warm_up+total_cnt_2*0.5];

        cout<<"========== model 1 box data: "<<model_1_low_whisker<<" "<<model_1_low_quartile<<" "\
        <<model_1_median<<" "<<model_1_high_quartile<<" "<<" "<<model_1_high_whisker<<endl;

        cout<<"========== model 2 box data: "<<model_2_low_whisker<<" "<<model_2_low_quartile<<" "\
        <<model_2_median<<" "<<model_2_high_quartile<<" "<<" "<<model_2_high_whisker<<endl;


        sort(model_1_wait_list.begin()+warm_up, model_1_wait_list.end());
        sort(model_2_wait_list.begin()+warm_up, model_2_wait_list.end());

        model_1_low_whisker = model_1_wait_list[warm_up+total_cnt_1*0.05];
        model_1_low_quartile = model_1_wait_list[warm_up+total_cnt_1*0.25];
        model_1_high_whisker = model_1_wait_list[warm_up+total_cnt_1*0.95];
        model_1_high_quartile = model_1_wait_list[warm_up+total_cnt_1*0.75];
        model_1_median = model_1_wait_list[warm_up+total_cnt_1*0.5];


        model_2_low_whisker = model_2_wait_list[warm_up+total_cnt_2*0.05];
        model_2_low_quartile = model_2_wait_list[warm_up+total_cnt_2*0.25];
        model_2_high_whisker = model_2_wait_list[warm_up+total_cnt_2*0.95];
        model_2_high_quartile = model_2_wait_list[warm_up+total_cnt_2*0.75];
        model_2_median = model_2_wait_list[warm_up+total_cnt_2*0.5];

        cout<<"========== model 1 box data add wait: "<<model_1_low_whisker<<" "<<model_1_low_quartile<<" "\
        <<model_1_median<<" "<<model_1_high_quartile<<" "<<" "<<model_1_high_whisker<<endl;

        cout<<"========== model 2 box data add wait: "<<model_2_low_whisker<<" "<<model_2_low_quartile<<" "\
        <<model_2_median<<" "<<model_2_high_quartile<<" "<<" "<<model_2_high_whisker<<endl;
        
       
        cout<<"model 1 avg latency: "<<model_1_avg_lat<<" model 2 avg latency: " <<model_2_avg_lat<<endl;
        cout<<"model 1 avg latency no wait: "<<model_1_avg_lat_no_wait<<" model 2 avg latency no wait:" <<model_2_avg_lat_no_wait<<endl;
        peak_tp = (default_stream_num * 1000) / max(model_1_median*0.5, model_2_median*0.5);
        cout <<" peak throughput:" << peak_tp << "request/s" <<endl;

        for(auto& co_lat : co_stream_list) {
            cout<<co_lat<<" ";
        }
        cout<<"\n";
    }

 private:
    GPUStatus _gpu_stats;

};
