#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <map>
#include "../src/backend/runtime.cpp"
#include "common.h"
using namespace std;

enum LayerKind {CONV, CONV_RELU, DEPTHWISECONV, MAXPOOL, DENSE, SLEEP, BATCH_MATMUL, TENSORADD};
enum BERTPhase {
    EnGenQ,
    EnGenK,
    EnGenV,
    DeGenQ,
    DeGenK,
    DeGenV,
    CQK,
    CSV,
    EnMH,
    EnFF,
    EnFF1,
    EnFF2,
    EnAdd1,
    EnAdd2,
    DeMH1,
    DeMH2,
    DeFF,
    DeFF1,
    DeFF2,
    DeAdd1,
    DeAdd2,
    DeAdd3

};


class BlockLatencyItem {
  public:
   BlockLatencyItem(int block, float latency) : _block(block), _latency(latency) {}

   int GetBlock() { return _block; }
   float GetLatency() { return _latency; }
  
  private:
   int _block;
   float _latency;
};


class Layer {
 public:
  virtual ~Layer()  {
    //cout<<"layer deconstruct"<<endl;
  };

  Layer(RUNTIME runtime=ROCM) : _runtime(runtime){}
  

  string GetCandidateNameById(size_t candidate_idx) {
    string ret = "candidate" + to_string(candidate_idx);
    return ret;
  }

  float* GenGPUData(size_t size) {
    size_t mem_size =  sizeof(float) * size;
    float* h = (float*)malloc(mem_size);
    
    RandomInit(h, size);
    auto device_api = GetBackendHandle(_runtime);
    auto d = static_cast<float*>(device_api->AllocDataSpace(size, DLFLOAT));
    device_api->CopyDataFromTo(h, d, mem_size, H2D);
    return d;
  }

  float* GenGPUDataAsync(size_t size, DLStream stream) {
    size_t mem_size =  sizeof(float) * size;
    float* h = (float*)malloc(mem_size);
    
    RandomInit(h, size);
    auto device_api = GetBackendHandle(_runtime);
    auto d = static_cast<float*>(device_api->AllocDataSpace(size, DLFLOAT));
    device_api->CopyDataFromToAsync(h, d, mem_size, H2D, stream);
    return d;
  }



  void RegisterCandidate(const string& model_name, const string& layer_idx_str, \
                        const string& candidate_name, size_t Block, size_t Thread, const string& layer_name="") {
    _candidate_list[candidate_name].push_back(Block);
    _candidate_list[candidate_name].push_back(Thread);
    _candidate_size_list.push_back(Block);
    auto device_api = GetBackendHandle(_runtime);
    _layer_name = layer_name;
    device_api->CompileKernel(model_name, layer_idx_str, candidate_name, layer_name);
    if (_runtime == ROCM) {
      auto function = static_cast<hipFunction_t>(device_api->GetFunction(model_name, layer_idx_str, candidate_name, layer_name));
      _function_list[candidate_name] = function;
    }
  }

  int GetLayerCandidateCnt() {
    return _function_list.size();
  }

  void GetLayerCandidateSizeList(vector<int>& list) {
    for(auto size : _candidate_size_list) {
      list.push_back(size);
    }
  }

  void AddBlockLatency(int block, float latency_us) {
    auto pair = make_shared<BlockLatencyItem>(block, latency_us);
    _block_latency_pair.push_back(pair);
  }
  

  virtual void Compute(string& candidate_name, DLStream stream) {}

  virtual unsigned long int GetComputation() {
    return 0;
  }

  LayerKind GetLayerKind() {
    return _layer_kind;
  }

  string GetLayerName() {
    return _layer_name;
  }

  /*
  
  string GetLayerName() {
    enum LayerKind {CONV, CONV_RELU, DEPTHWISECONV, MAXPOOL, DENSE, SLEEP, BATCH_MATMUL, TENSORADD};
    string ret = "";
    if (_layer_kind == CONV || _layer_kind == CONV_RELU) {
      ret = "CONV";
    } else if(_layer_kind == DEPTHWISECONV) {
      ret = "DEPTHWISECONV";
    } else if (_layer_kind == MAXPOOL) {
      ret = "MAXPOOL";
    } else if (_layer_kind == DENSE) {
      ret = "DENSE";
    } else if (_layer_kind == BATCH_MATMUL) {
      ret = "BATCH_MATMUL";
    } 
    return ret;
  }
  */
  

 public:
  unordered_map<string, vector<size_t>> _candidate_list;
  unordered_map<string, hipFunction_t> _function_list;
  vector<int> _candidate_size_list;
  vector<shared_ptr<BlockLatencyItem>> _block_latency_pair; // first item is block of candiate, second item is latency corresponding block
  RUNTIME _runtime;
  LayerKind _layer_kind;
  string _layer_name;

 private:
  DLStream _stream;

};

#endif