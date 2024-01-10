#ifndef DK_DK_CO_CONFIG_H
#define DK_DK_CO_CONFIG_H

#include "co_config/co_config.h"

class DKCC : public CoConfig {
 public:
  void ConstructOfflineResult() {
    _layer_idx_1 = {0,9,13,15,19,20,21,23};
    _layer_idx_2 = {0,9,13,15,19,20,21,23};

    _layer_1_candidates = {0,0,1,0,1,0,1,0};
    _layer_2_candidates = {0,0,1,1,1,1,1,0};

    if (_layer_idx_1.size() != _layer_idx_2.size() || _layer_1_candidates.size() != _layer_2_candidates.size()\
    || _layer_1_candidates.size() != _layer_idx_1.size()) {
      cout<<"Co Config Not Match"<<endl;
      exit(-1);
    }
  }
   
};

#endif