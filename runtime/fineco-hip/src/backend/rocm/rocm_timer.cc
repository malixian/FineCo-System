#ifndef ROCM_TIMER_H
#define ROCM_TIMER_H

#include "hip/hip_runtime_api.h"
#include "backend/timer.h"

class ROCMTimer : public Timer {
public:
    ROCMTimer(DLStream stream) {
		stream_ = static_cast<hipStream_t>(stream);
		HIP_CALL(hipEventCreate(&start_));
		HIP_CALL(hipEventCreate(&stop_));
    }

	void Start() {
		HIP_CALL(hipEventRecord(start_, stream_));
	}

	void Stop() {
		HIP_CALL(hipEventRecord(stop_, stream_)); 
	}

	float SyncAndGetElapsedms() {
		HIP_CALL(hipEventSynchronize(stop_));
		float milliseconds = 0;
		HIP_CALL(hipEventElapsedTime(&milliseconds, start_, stop_));
		return milliseconds;
	}

	~ROCMTimer() {
		HIP_CALL(hipEventDestroy(start_));
		HIP_CALL(hipEventDestroy(stop_));
	}

private:
	hipEvent_t start_;
	hipEvent_t stop_;
	hipStream_t stream_;

};

#endif