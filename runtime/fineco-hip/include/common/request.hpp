#ifndef REQUEST_H
#define REQUEST_H

enum ModelKind {
    M1, M2, EMPTY
};

enum QoSKind {
    LowQoS,
    HighQoS,
    Empty,
};


class Request {
 public:
    Request(ModelKind model_kind, QoSKind qos_kind, int gene_timestep) : _model_kind(model_kind), _qos_kind(qos_kind), _gene_timestep(gene_timestep){}
    void SetSubmitTime(int64_t submit_time) { _submit_time = submit_time; }
    int64_t GetSubmitTime() {return _submit_time;}

    void SetWaitTime(int64_t wait_time) { _wait_time = wait_time; }
    int64_t GetWaitTime() {return _wait_time;}

    QoSKind GetQoSKind() {return _qos_kind;}
    ModelKind GetModelKind() {return _model_kind;}
    int GetGeneTimestep() {return _gene_timestep;}

    void SetProgress(float progress) {_progress_bar = progress;}
    float GetProgress() {return _progress_bar;}

    void SetAllocatedSM(int sm) {_used_sm = sm;}
    int GetUsedSM() {return _used_sm;}

    void SetEstimatedLatency(float latency_ms) {
        _est_latency_ms = latency_ms;
    }
    int GetEstimatedLatency() {
       int latency_us = int(_est_latency_ms * 1000);
        return latency_us;
    }

 private:
    ModelKind _model_kind;
    QoSKind _qos_kind;
    int64_t _submit_time;
    int64_t _wait_time;
    int _gene_timestep;
    float  _progress_bar = 0;
    int _used_sm;
 public:
    float _est_latency_ms = 0;
};


#endif