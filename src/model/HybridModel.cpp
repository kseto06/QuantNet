#include "HybridModel.h"
#include <cmath>
#include <vector>

namespace HybridModel {
    double MSE(const std::vector<double>& pred, const std::vector<double>& target) {
        double loss = 0.0;
        for (size_t i = 0; i < pred.size(); i++) {
            loss += std::pow(pred[i] - target[i], 2);
        }
        return loss/pred.size();
    }
}