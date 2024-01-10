#include "service/service.h"
#include "common/thread_pool.hpp"
#include "common/gpu_stats.hpp"
#include <algorithm>


class FineCoService : public PairService {
 public: 

    void Accept() {
        cout<<"==== Start FineCo Service ========"<<endl;

        vector<int> candidate_cnt_1;
        _model_1->GetAllLayerCandidateCnt(candidate_cnt_1);
        vector<int> candidate_high_selector_1;
        ServiceGenHighQoSCombination(candidate_cnt_1, candidate_high_selector_1);
        vector<int> candidate_low_selector_1;
        ServiceGenLowQoSCombination(candidate_cnt_1, candidate_low_selector_1);

         vector<int> candidate_cnt_2;
        _model_2->GetAllLayerCandidateCnt(candidate_cnt_2);
        vector<int> candidate_high_selector_2;
        ServiceGenHighQoSCombination(candidate_cnt_2, candidate_high_selector_2);
        vector<int> candidate_low_selector_2;
        ServiceGenLowQoSCombination(candidate_cnt_2, candidate_low_selector_2);

        int default_stream_num = 2;
        int high_priority_stream_num = 0;
        int pool_size = 10;
        ThreadPool client_pool(pool_size);

        
        int stream_num = default_stream_num;
        vector<float*> _output_memory_pool_1(stream_num);
        vector<float*> _input_memory_pool_1(stream_num);
        vector<float*> _output_memory_pool_2(stream_num);
        vector<float*> _input_memory_pool_2(stream_num);

        vector<DLStream> _stream_pool(default_stream_num);


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
        
        
        int model_1_number = 0;
        int model_2_number = 0;

        // flexible sycn client threads
        vector<int> available_streams;
        for(int i=0; i<default_stream_num; i++) {
            available_streams.push_back(i);
        }

        int acc_model1_number = 0;
        unordered_map<int, future<float>> results;

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

    
        float model_1_avg_lat = 0.0;
        float model_2_avg_lat = 0.0;
        int high_qos_model_1_cnt = 0;
        int high_qos_model_2_cnt = 0;
        float model_1_avg_lat_no_wait = 0.0;
        float model_2_avg_lat_no_wait = 0.0;
        int64_t first_submit_timestamp = 0;
        int arrival_interval_us = 20000;
        float avg_wait_time = 0;

        int total_reqs = _total_request_queue.size();
        int total_idle_time = arrival_interval_us * total_reqs / 1000; 

        auto start = system_clock::now();
        while(_total_request_queue.size() > 0) {
            // concurrent submit request by stream number
            //cout<<"left request size: "<<_high_request_queue.size()<<endl;
            int submit_number = 0;
            /*
            vector<Request> request_group;
            auto ts = _high_request_queue[0]->GetGeneTimestep();
            int idx = 0;
            while(_high_request_queue.size() > 0) {
                if(_high_request_queue[idx]->GetGeneTimestep() == ts) {
                    request_group.emplace_back(_high_request_queue[idx]);
                }
            }
            */
            while (available_streams.size()>0 && _total_request_queue.size()>0) {
                int stream_id = available_streams[0];
                auto stream = _stream_pool[stream_id];
                auto cur_request = _total_request_queue[0];
                bool has_submit = false;
                auto submit_time = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count();
                if (first_submit_timestamp == 0) first_submit_timestamp = submit_time;
                auto arrivel_time = cur_request->GetGeneTimestep()*arrival_interval_us + first_submit_timestamp;
                if (cur_request->GetModelKind() == M1 && submit_time > arrivel_time) {
                    submit_number++;
                    has_submit = true;
                    if (cur_request->GetQoSKind() == HighQoS) {
                        results[stream_id] = client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_high_selector_1), _output_memory_pool_1[stream_id],\
                        _input_memory_pool_1[stream_id], input_mem_size_1, stream, true);
                        high_qos_model_1_cnt++;
                    } else if (cur_request->GetQoSKind() == LowQoS) {
                        results[stream_id] = client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_low_selector_1), _output_memory_pool_1[stream_id],\
                        _input_memory_pool_1[stream_id], input_mem_size_1, stream, true);
                    }
                    model_1_number++;
                } else if (cur_request->GetModelKind() == M2 && submit_time > arrivel_time ) {
                    submit_number++;
                    has_submit = true;
                    if (cur_request->GetQoSKind() == HighQoS) {
                        results[stream_id] = client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_high_selector_2), _output_memory_pool_2[stream_id],\
                        _input_memory_pool_2[stream_id], input_mem_size_2, stream, true);
                        high_qos_model_2_cnt++;
                    } else if(cur_request->GetQoSKind() == LowQoS) {
                        results[stream_id] = client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_low_selector_2), _output_memory_pool_2[stream_id],\
                        _input_memory_pool_2[stream_id], input_mem_size_2, stream, true);
                    }
                    model_2_number++;
                }
                
                if(has_submit) {
                    cur_request->SetSubmitTime(submit_time);
                    cur_request->SetWaitTime((submit_time - arrivel_time)/1000);
                    //cout<<"submit time: "<<submit_time<<" arrivel time:"<<arrivel_time<<" "<<endl;
                    //cout<<"set stream id: "<<stream_id<<endl;
                    _gpu_stats.Update(stream_id, cur_request);
                    _total_request_queue.pop_front();
                    // free available stream
                    for(int idx=0; idx < available_streams.size(); idx++) {
                        if (available_streams[idx] == stream_id) {
                            available_streams.erase(available_streams.begin()+idx);
                        }
                    }
                }
            }

            // if(submit_number > 0)
            //    cout<<"avilable stream size: "<<available_streams.size()<<" request queue.size(): "<<_high_request_queue.size()<<endl;

            //cout<<"gpu running streams : " << _gpu_stats._request_stats.size()<<endl;
            //cout<<"gpu available streams : " << available_streams.size()<<endl;

            // *** async based time diff
            for(auto itor = _gpu_stats._request_stats.begin(); itor != _gpu_stats._request_stats.end(); ) {
                auto stats = *itor;
                auto cur_time = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count();
                auto submit_time = stats->GetSubmitTime();
                auto diff_time = cur_time - submit_time;
                auto stream_id = stats->GetStreamId();
                auto wait_time = stats->GetWaitTime();
                //cout<<"wait time: "<<wait_time<<endl;
                //cout<<"try stream id: "<<stream_id<<endl;
                //cout<<"cur model is: "<<stats->GetModelKind()<<" diff time: "<<diff_time<<endl;
                if ((stats->GetModelKind() == M1 && diff_time > model_1_latency)\
                 || (stats->GetModelKind() == M2 && diff_time > model_2_latency)) {
                    available_streams.push_back(stats->GetStreamId());
                    //cout<<"stream id: "<<stream_id;
                    auto lat = results[stream_id].get();
                    //cout<<" lat:"<<lat<<endl;
                    avg_wait_time += wait_time;
                    if (stats->GetModelKind() == M1 && stats->GetQoSKind() == HighQoS) {
                        model_1_avg_lat += lat+wait_time;
                        model_1_avg_lat_no_wait += lat;
                        model_1_lat_list.push_back(lat);
                        //cout<<"high qos model 1 lat: "<<lat<<endl;
                    } else if (stats->GetModelKind() == M2 && stats->GetQoSKind() == HighQoS) {
                        model_2_avg_lat += lat+wait_time;
                        model_2_avg_lat_no_wait += lat;
                        model_2_lat_list.push_back(lat);
                        //cout<<"high qos model 2 lat: "<<lat<<endl;
                    }
                    _gpu_stats._request_stats.erase(itor);
                    //cout<<"erase itor"<<endl;
                } else {
                    itor++;
                }
            }
            
        }

        int warm_up = 0;

        sort(model_1_lat_list.begin()+warm_up, model_1_lat_list.end());
        sort(model_2_lat_list.begin()+warm_up, model_2_lat_list.end());

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

        int total_cnt_1 = high_qos_model_1_cnt;
        int total_cnt_2 = high_qos_model_2_cnt;
        int all_cnt = total_cnt_1 + total_cnt_2;

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

        cout << "model 1 avg latency: " << model_1_avg_lat / high_qos_model_1_cnt << " model 2 avg latency: " << model_2_avg_lat / high_qos_model_2_cnt <<endl;
        cout << "model 1 avg latency no wait: " << model_1_avg_lat_no_wait / high_qos_model_1_cnt << " model 2 avg latency no wait: " << model_2_avg_lat_no_wait / high_qos_model_1_cnt <<endl;
        
        auto end   = system_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        float throughput = (model_1_number +  model_2_number)  * 1000 / (duration - total_idle_time) ;

        cout << "run cost " << duration  - total_idle_time << "ms" << " throughput:" << throughput << "reqs/s" <<endl;

        float peak_tp = ((default_stream_num + high_priority_stream_num) * 1000) / (model_1_median*0.5 + model_2_median*0.5);
        cout<<"peak throughput: "<<peak_tp<<endl;
    }

 private:
    GPUStatus _gpu_stats;

};
