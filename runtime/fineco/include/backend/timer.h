#ifndef TIMER_H
#define TIMER_H


class Timer {
 public:

  virtual void Start() = 0;

  virtual void Stop() = 0;

  virtual float SyncAndGetElapsedms() = 0;

};

#endif 