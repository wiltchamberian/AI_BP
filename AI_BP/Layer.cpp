#include "Layer.h"

Layer operator + (const Layer& l1, const Layer& l2) {
  Layer layer(l1.weights.size(), l1.weights[0].size());
  for (int i = 0; i < l1.weights.size(); ++i) {
    for (int j = 0; j < l1.weights[i].size(); ++j) {
      layer.weights[i][j] = l1.weights[i][j] + l2.weights[i][j];
    }
    layer.b[i] = l1.b[i] + l2.b[i];
  }
  return layer;
}

Layer& Layer::operator /= (int n) {
  for (int i = 0; i < weights.size(); ++i) {
    for (int j = 0; j < weights[i].size(); ++j) {
      weights[i][j] = weights[i][j] / n;
    }
    b[i] = b[i] / n;
  }
  return *this;
}

