#ifndef R50_R50_CO_CONFIG_H
#define R50_R50_CO_CONFIG_H

#include "co_config/co_config.h"

class R50R50CC : public CoConfig {
 public:
  void ConstructOfflineResult() {
    _layer_idx_1 = {3,6,9,15,18,21,23,24,25,26,29,31,32,35,37,38,40};
    _layer_idx_2 = {3,6,9,15,18,21,23,24,25,26,29,31,32,35,37,38,40};

    _layer_1_candidates = {0,0,0,1,1,1,0,0,2,0,1,1,0,0,1,1,1};
    _layer_2_candidates = {0,0,0,1,1,1,0,0,1,0,1,2,0,0,2,1,2};

    if (_layer_idx_1.size() != _layer_idx_2.size() || _layer_1_candidates.size() != _layer_2_candidates.size()\
    || _layer_1_candidates.size() != _layer_idx_1.size()) {
      cout<<"Co Config Not Match"<<endl;
      exit(-1);
    }
  }
   
};

#endif