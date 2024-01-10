#ifndef GPU_STATS_H
#define GPU_STATS_H

#include <unordered_map>
#include <utility>
#include "common/request.hpp"

class RequestStatus {
 public:
    RequestStatus(int stream_id, shared_ptr<Request> request) : _stream_id(stream_id), _request(request) {}

    ModelKind GetModelKind() {return _request->GetModelKind();}
    int GetStreamId() {return _stream_id;}
    int64_t GetSubmitTime() {return _request->GetSubmitTime();}
    int64_t GetWaitTime() {return _request->GetWaitTime();}
    QoSKind GetQoSKind() {return _request->GetQoSKind();}
    int GetUsedSM() {return _request->GetUsedSM();}
    int GetEstimatedLatency() {return _request->GetEstimatedLatency();}
    void SetSync() {_has_sync = true;}
    bool GetHasSync() {return _has_sync;}

 private:
    shared_ptr<Request> _request;
    int _stream_id;
    bool _has_sync = false;
};

class GPUStatus {
public:

    void Update(int stream_id, shared_ptr<Request> request) {
        auto rs = make_shared<RequestStatus>(stream_id, request);
        _request_stats.push_back(rs);

        _stream_mapping[stream_id].push_back(rs);
        _stream_runtime[stream_id] += request->GetEstimatedLatency();
    }

    int GetRequestSize() {
        return _request_stats.size();
    }

    bool ExistHighQoS() {
        for (auto rs: _request_stats) {
            if (rs->GetQoSKind() == HighQoS) {
                return true;
            }
        }
        return false;
    }

    bool HasHighQoSDoneById(int stream_idx) {
        for (auto &rs : _stream_mapping[stream_idx]) {
            if (rs->GetQoSKind() == HighQoS && !rs->GetHasSync()) {
                return false;
            }
        }
        return true;
    }

    bool HasDoneById(int stream_idx) {
        for (auto &rs : _stream_mapping[stream_idx]) {
            if (!rs->GetHasSync()) {
                return false;
            }
        }
        return true;
    }

    bool AllHighQoSDone() {
        for (auto &item : _stream_mapping) {
            for (auto &rs : item.second) {
                if (rs->GetQoSKind() == HighQoS && !rs->GetHasSync()) {
                    return false;
                }
            }
        }
        return true;
    }

    // Get need sycn stream
    int GetNeedSyncStreamId() {
        int stream_id = -1;
        int min_latency = -1;
        for (const auto &item : _stream_runtime) {
            //cout<<"stream idx"<<item.first<<" "<<"acc latency: "<<item.second<<endl;
            if (HasDoneById(item.first)) {
            //    cout<<"stream no need sync: "<<item.first<<endl;
                continue;
            }
            if (min_latency == -1) {
                min_latency = item.second;
                stream_id = item.first;
            } else {
                if (min_latency > item.second) {
                    min_latency = item.second;
                    stream_id = item.first; 
                }
            }
        }

        return stream_id;
    }

    int GetFillStreamId() {
        int stream_id = 0;
        int min_latency = -1;
        for (const auto &item : _stream_runtime) {
            stream_id = item.first; 
            //cout<<"stream idx"<<item.first<<" "<<"acc latency: "<<item.second<<endl;
            if (min_latency == -1) {
                min_latency = item.second;
                stream_id = stream_id;
            } else {
                if (min_latency > item.second) {
                    min_latency = item.second;
                    stream_id = item.first; 
                }
            }
        }
        return stream_id;
    }


    int GetStreamStartLatency() {
        int diff_latency = 0;
        int min_latency = 0;
        int max_latency = 0;
        for (const auto &item : _stream_runtime) {
            if (min_latency == -1) {
                min_latency = item.second;
                max_latency = item.second;
            } else {
                if (min_latency > item.second) {
                    min_latency = item.second;
                }
            }
        }
        return min_latency;
    }

    void ClearStreamRuntime() {
        _request_stats.clear();
        _stream_mapping.clear();
        _stream_runtime.clear();
    }

    int getLeftSM(vector<pair<int64_t, int>>& tmp_queue_end_ts, int idx) {
        int available_sm = 108;
        for(int i=idx; i<tmp_queue_end_ts.size(); i++) {
            available_sm -= tmp_queue_end_ts[i].second; 
        }
        if (available_sm == 0) return 0;
        else return available_sm;
    }

    void GetCurrentAvailableSmSequence(int concurrent_cnt, int64_t cur_submit_timestamp, vector<pair<int, int>>& available_sm_seq) {
        
        vector<pair<int64_t, int>> tmp_queue_end_ts;
        for(int rid = 0; rid < _request_stats.size(); rid++) {
            auto request = _request_stats[rid];
            int64_t queue_est_finished_timestamp = request->GetSubmitTime() + request->GetEstimatedLatency();
            int used_sm = request->GetUsedSM();
            pair<int64_t, int> tmp_pair = make_pair(queue_est_finished_timestamp, used_sm);
            tmp_queue_end_ts.emplace_back(tmp_pair);
        }
        sort(tmp_queue_end_ts.begin(), tmp_queue_end_ts.end(), [](const pair<int, int64_t>& lhs, const pair<int, int64_t>& rhs){return lhs.first < rhs.first;});

        // construct left sm status
        int64_t last_end_timestamp = cur_submit_timestamp;
        for(int i=0; i<tmp_queue_end_ts.size(); i++) {
            auto item = tmp_queue_end_ts[i];
            int interval = item.first - last_end_timestamp;
            last_end_timestamp = item.first;
            pair<int, int> tmp_pair = make_pair(interval, getLeftSM(tmp_queue_end_ts, i));
            available_sm_seq.emplace_back(tmp_pair);
        }
        if (available_sm_seq.size() == 0) {
            int all_available_sm = 108;
            pair<int, int> tmp_pair = make_pair(-1, all_available_sm);
            available_sm_seq.emplace_back(tmp_pair);
        }
    }

    /*
    void GetCurrentAvailableSmSequenceByEstLat(int concurrent_cnt, int64_t cur_submit_timestamp, float cur_est_latency_ms, vector<pair<int, int>>& available_sm_seq) {
        
        int cur_est_latency_us = int(cur_est_latency_ms * 1000);
        vector<pair<int64_t, int>> tmp_queue_end_ts;
        for(int rid = 0; rid < _request_stats.size(); rid++) {
            auto request = _request_stats[rid];
            int64_t queue_est_finished_timestamp = request->GetSubmitTime() + request->GetEstimatedLatency();
            int used_sm = request->GetUsedSM();
            pair<int64_t, int> tmp_pair = make_pair(queue_est_finished_timestamp, used_sm);
            tmp_queue_end_ts.emplace_back(tmp_pair);
        }
        sort(tmp_queue_end_ts.begin(), tmp_queue_end_ts.end(), [](const pair<int64_t, int>& lhs, const pair<int64_t, int>& rhs){return lhs.second < rhs.second;});

        // construct left sm status
        vector<pair<int, int>> lef_sm_status;
        int64_t last_end_timestamp = 0;
        for(int i=0; i<tmp_queue_end_ts.size(); i++) {
            auto item = tmp_queue_end_ts[i];
            int interval = item.first - cur_submit_timestamp - last_end_timestamp;
            last_end_timestamp = item.first;
            pair<int, int> tmp_pair = make_pair(interval, getLeftSM(tmp_queue_end_ts, i));
            lef_sm_status.emplace_back(tmp_pair);
        }

        int cur_est_latency_us_copy = cur_est_latency_us;
        int last_percentage = 0;
        for(int i=0; i<lef_sm_status.size(); i++) {
            auto stats_pair = lef_sm_status[i];
            if(cur_est_latency_us_copy >= stats_pair.first) {
                int cur_percentage = stats_pair.first / cur_est_latency_us_copy;
                int next_percentage = cur_percentage + last_percentage;
                if (next_percentage >= 100) next_percentage = 100;
                pair<int, int> ret_pair = make_pair(next_percentage, stats_pair.second / concurrent_cnt);
                available_sm_seq.emplace_back(ret_pair);
                last_percentage = next_percentage;
                cur_est_latency_us_copy -= stats_pair.first;
            } else {
                pair<int, int> ret_pair = make_pair(100, stats_pair.second / concurrent_cnt);
                last_percentage = 100;
                available_sm_seq.emplace_back(ret_pair);
                return ;
            }
        }
        int all_sm_cnt = 108;
        if (last_percentage < 100) {
            pair<int, int> ret_pair = make_pair(100, all_sm_cnt / concurrent_cnt);
            available_sm_seq.emplace_back(ret_pair);
            return;
        }
        return ;
        
    }
    */


    vector<shared_ptr<RequestStatus>> _request_stats;
    unordered_map<int, vector<shared_ptr<RequestStatus>>> _stream_mapping;
    unordered_map<int, int> _stream_runtime;
};


#endif