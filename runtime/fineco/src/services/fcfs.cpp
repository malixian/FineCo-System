#include "service/service.h"
#include "common/thread_pool.hpp"
#include "common/gpu_stats.hpp"
#include <algorithm>

class FCFSService : public PairService {
 public: 

    void Accept() {
        cout<<"==== Start FCFS Service ========"<<endl;

        vector<int> candidate_cnt_1;
        _model_1->GetAllLayerCandidateCnt(candidate_cnt_1);
        vector<int> candidate_high_selector_1;
        ServiceGenDefaultCombination(candidate_cnt_1, candidate_high_selector_1);
        vector<int> candidate_low_selector_1;
        ServiceGenDefaultCombination(candidate_cnt_1, candidate_low_selector_1);

         vector<int> candidate_cnt_2;
        _model_2->GetAllLayerCandidateCnt(candidate_cnt_2);
        vector<int> candidate_high_selector_2;
        ServiceGenDefaultCombination(candidate_cnt_2, candidate_high_selector_2);
        vector<int> candidate_low_selector_2;
        ServiceGenDefaultCombination(candidate_cnt_2, candidate_low_selector_2);

        int default_stream_num = 1;
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

        vector<float> model_1_lat_list_rt;
        vector<float> model_2_lat_list_rt;

        vector<float> model_1_lat_list_nowait_rt;
        vector<float> model_2_lat_list_nowait_rt;

        vector<float> model_1_lat_list_be;
        vector<float> model_2_lat_list_be;

        vector<float> model_1_wait_list;
        vector<float> model_2_wait_list;

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
        
        //int64_t first_submit_timestamp = 0;
        //int arrival_interval_us = 20000;
        float avg_wait_time = 0;
        vector<float> wait_time_1;
        vector<float> wait_time_2;
        float peak_tp = 0;
        int peak_reqs = 0;

        auto begin_ts = chrono::steady_clock::now();

        int total_requests_cnt = _total_request_queue.size();

        //auto start = system_clock::now();
        while(_total_request_queue.size() > 0) {
            //usleep(500);
            // concurrent submit request by stream number
            //cout<<"left request size: "<<_total_request_queue.size()<<endl;
            int submit_number = 0;
            int first_timestamp = _total_request_queue[0]->GetGeneTimestep();
            int stream_id = 0;
            int concurrent_num = default_stream_num;

            int high_qos_cnt_1 = 0;
            int low_qos_cnt_1 = 0;
            int high_qos_cnt_2 = 0;
            int low_qos_cnt_2 = 0;


            vector<shared_ptr<Request>> tmp_req;
            for(int ri=0; ri<_total_request_queue.size(); ri++) {
                if(_total_request_queue[ri]->GetGeneTimestep() == first_timestamp) {
                    tmp_req.push_back(_total_request_queue[ri]);
                }
            }
            auto arrival_timestamp = chrono::steady_clock::now();
            for(auto cur_request : tmp_req) {
                if (stream_id >= default_stream_num) stream_id = stream_id % default_stream_num;
                auto stream = _stream_pool[stream_id];
                vector<future<float>> results_model1_rt;
                vector<future<float>> results_model1_be;
                vector<future<float>> results_model2_rt;
                vector<future<float>> results_model2_be;
                //if (first_submit_timestamp == 0) first_submit_timestamp = submit_time;
                //auto arrivel_time = cur_request->GetGeneTimestep()*arrival_interval_us + first_submit_timestamp; 
                if (cur_request->GetModelKind() == M1 && cur_request->GetQoSKind() != Empty) {
                    submit_number++;
                    if (cur_request->GetQoSKind() == HighQoS) {
                        results_model1_rt.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_high_selector_1), _output_memory_pool_1[stream_id],\
                        _input_memory_pool_1[stream_id], input_mem_size_1, stream, true));
                        high_qos_cnt_1++;
                    } else {
                        results_model1_be.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_low_selector_1), _output_memory_pool_1[stream_id],\
                        _input_memory_pool_1[stream_id], input_mem_size_1, stream, true));
                        low_qos_cnt_1++;
                    }
                    model_1_number++;
                    stream_id++;
                    _total_request_queue.pop_front();
                    //cout<<"request size: "<<_total_request_queue.size()<<endl;
                } else if (cur_request->GetModelKind() == M2 && cur_request->GetQoSKind() != Empty) {
                    submit_number++;
                    if (cur_request->GetQoSKind() == HighQoS) {
                        results_model2_rt.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_high_selector_2), _output_memory_pool_2[stream_id],\
                        _input_memory_pool_2[stream_id], input_mem_size_2, stream, true));
                        high_qos_cnt_2++;
                    } else {
                        results_model2_be.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_low_selector_2), _output_memory_pool_2[stream_id],\
                        _input_memory_pool_2[stream_id], input_mem_size_2, stream, true));
                        low_qos_cnt_2++;
                    }
                    model_2_number++;
                    stream_id++;
                    _total_request_queue.pop_front();
                }

                for(int i=0; i<results_model1_rt.size(); i++) {
                    auto r = results_model1_rt[i].get();
                    auto finish_timestamp = chrono::steady_clock::now();
                    float duration = chrono::duration_cast<chrono::microseconds>(finish_timestamp - arrival_timestamp).count() * 1.0 / 1000;
                    float wait_time = duration - r;
                    if (wait_time < 0) {
                        cout<<duration<<" "<<r<<endl;
                        cout<<"============= exception, wait time should greater than duration ============"<<endl;
                    }
                    model_1_wait_list.push_back(wait_time);
                    model_1_lat_list_rt.push_back(duration);
                    model_1_lat_list_nowait_rt.push_back(r);
                }

                for(int i=0; i<results_model2_rt.size(); i++) {
                    auto r = results_model2_rt[i].get();
                    auto finish_timestamp = chrono::steady_clock::now();
                    float duration = duration_cast<microseconds>(finish_timestamp - arrival_timestamp).count() * 1.0 / 1000;
                    float wait_time = duration - r;
                    if (wait_time < 0) {
                        cout<<duration<<" "<<r<<endl;
                        cout<<"============= exception, wait time should greater than duration ============"<<endl;
                    }
                    model_2_wait_list.push_back(wait_time);
                    model_2_lat_list_rt.push_back(duration);
                    model_2_lat_list_nowait_rt.push_back(r);
                }

                for(int i=0; i<results_model1_be.size(); i++) {
                    auto r = results_model1_be[i].get();
                    auto finish_timestamp = chrono::steady_clock::now();
                    float duration = duration_cast<microseconds>(finish_timestamp - arrival_timestamp).count() * 1.0 / 1000;
                    float wait_time = duration - r;
                    if (wait_time < 0) {
                        cout<<duration<<" "<<r<<endl;
                        cout<<"============= exception, wait time should greater than duration ============"<<endl;
                    }
                    model_1_wait_list.push_back(wait_time);
                    model_1_lat_list_be.push_back(duration);
                }

                for(int i=0; i<results_model2_be.size(); i++) {
                    auto r = results_model2_be[i].get();
                    auto finish_timestamp = chrono::steady_clock::now();
                    float duration = duration_cast<microseconds>(finish_timestamp - arrival_timestamp).count() * 1.0 / 1000;
                    float wait_time = duration - r;
                    if (wait_time < 0) {
                        cout<<duration<<" "<<r<<endl;
                        cout<<"============= exception, wait time should greater than duration ============"<<endl;
                    }
                    model_2_wait_list.push_back(wait_time);
                    model_2_lat_list_be.push_back(duration);
                }
            }
        }
        
        auto end_ts = chrono::steady_clock::now();
        float total_dur = duration_cast<microseconds>(end_ts - begin_ts).count();
        float tp = (total_requests_cnt / total_dur) * 1e+6 ;
        cout <<" throughput:" << tp << "request/s" <<endl;
        
        float avg_wait_time_1 = 0;
        float avg_wait_time_2 = 0;

        int total_cnt_1_rt = model_1_lat_list_rt.size();
        int total_cnt_2_rt = model_2_lat_list_rt.size();
        int total_cnt_1_be = model_1_lat_list_be.size();
        int total_cnt_2_be = model_2_lat_list_be.size();

        cout<<total_cnt_1_rt<<" "<<total_cnt_2_rt<<" "<<total_cnt_1_be<<" "<<total_cnt_2_be<<endl;
        
        for (auto wait_time : model_1_wait_list) {
            avg_wait_time_1 += wait_time;
        }
        avg_wait_time_1 /= model_1_wait_list.size();

        for (auto wait_time : model_2_wait_list) {
            avg_wait_time_2 += wait_time;
        }
        avg_wait_time_2 /= model_2_wait_list.size();
        cout<<"model 1 avg wait latency : "<<avg_wait_time_1<<" model 2 avg wait latency: "<<avg_wait_time_2<<endl;
        
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
        
        sort(model_1_lat_list_rt.begin(), model_1_lat_list_rt.end());
        sort(model_2_lat_list_rt.begin(), model_2_lat_list_rt.end());

        model_1_low_whisker = model_1_lat_list_rt[total_cnt_1_rt*0.05];
        model_1_low_quartile = model_1_lat_list_rt[total_cnt_1_rt*0.25];
        model_1_high_whisker = model_1_lat_list_rt[total_cnt_1_rt*0.95];
        model_1_high_quartile = model_1_lat_list_rt[total_cnt_1_rt*0.75];
        model_1_median = model_1_lat_list_rt[total_cnt_1_rt*0.5];


        model_2_low_whisker = model_2_lat_list_rt[total_cnt_2_rt*0.05];
        model_2_low_quartile = model_2_lat_list_rt[total_cnt_2_rt*0.25];
        model_2_high_whisker = model_2_lat_list_rt[total_cnt_2_rt*0.95];
        model_2_high_quartile = model_2_lat_list_rt[total_cnt_2_rt*0.75];
        model_2_median = model_2_lat_list_rt[total_cnt_2_rt*0.5];

        cout<<"========== model 1 box data: "<<model_1_low_whisker<<" "<<model_1_low_quartile<<" "\
        <<model_1_median<<" "<<model_1_high_quartile<<" "<<" "<<model_1_high_whisker<<endl;

        cout<<"========== model 2 box data: "<<model_2_low_whisker<<" "<<model_2_low_quartile<<" "\
        <<model_2_median<<" "<<model_2_high_quartile<<" "<<" "<<model_2_high_whisker<<endl;

        int warm_up = 0;
        sort(model_1_lat_list_nowait_rt.begin(), model_1_lat_list_nowait_rt.end());
        sort(model_2_lat_list_nowait_rt.begin(), model_2_lat_list_nowait_rt.end());
        model_1_low_whisker = model_1_lat_list_nowait_rt[warm_up+total_cnt_1_rt*0.05];
        model_1_low_quartile = model_1_lat_list_nowait_rt[warm_up+total_cnt_1_rt*0.25];
        model_1_high_whisker = model_1_lat_list_nowait_rt[warm_up+total_cnt_1_rt*0.95];
        model_1_high_quartile = model_1_lat_list_nowait_rt[warm_up+total_cnt_1_rt*0.75];
        model_1_median = model_1_lat_list_nowait_rt[warm_up+total_cnt_1_rt*0.5];

        if (total_cnt_2_rt > 0) {
            model_2_low_whisker = model_2_lat_list_nowait_rt[warm_up+total_cnt_2_rt*0.05];
            model_2_low_quartile = model_2_lat_list_nowait_rt[warm_up+total_cnt_2_rt*0.25];
            model_2_high_whisker = model_2_lat_list_nowait_rt[warm_up+total_cnt_2_rt*0.95];
            model_2_high_quartile = model_2_lat_list_nowait_rt[warm_up+total_cnt_2_rt*0.75];
            model_2_median = model_2_lat_list_nowait_rt[warm_up+total_cnt_2_rt*0.5];
        }

        cout<<"========== model 1 no wait "<< _model_1->GetModelName() <<" box data: "<<model_1_low_whisker<<" "<<model_1_low_quartile<<" "\
        <<model_1_median<<" "<<model_1_high_quartile<<" "<<" "<<model_1_high_whisker<<endl;

        cout<<"========== model 2 no wait "<< _model_2->GetModelName()<<" box data: "<<model_2_low_whisker<<" "<<model_2_low_quartile<<" "\
        <<model_2_median<<" "<<model_2_high_quartile<<" "<<" "<<model_2_high_whisker<<endl;
    }

 private:
    GPUStatus _gpu_stats;

};
