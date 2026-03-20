#pragma once
#include "network.h"

class SGD {
private:
    double learning_rate_;

public:
    explicit SGD(double lr);
    void step(Network& network);
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    double get_learning_rate() const { return learning_rate_; }
};
