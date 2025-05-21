#include "NeuralNetwork.h"
#include <cmath>

double NeuralNetwork::activate(double x) {
  //return 1.0 / (1 + exp(-x));
  return x > 0 ? x : 0.0;
}

double NeuralNetwork::dActivate(double x) {
  return x > 0 ? 1.0 : 0.0;
}

std::vector<double> NeuralNetwork::dC_da(std::vector<double>& a, std::vector<double>& y) {
  std::vector<double> output(a.size(), 0);
  for (int i = 0; i < a.size(); ++i) {
    output[i] = a[i] - y[i];
  }
  return output;
}

std::vector<double> NeuralNetwork::dsigma_dz(std::vector<double>& z) {
  std::vector<double> b(z.size());
  for (int i = 0; i < z.size(); ++i) {
    b[i] = dActivate(z[i]);
  }
  return b;
}

std::vector<std::vector<double>> NeuralNetwork::dC_dw(Sample& a, Sample& delta) {
  std::vector<std::vector<double>> w(delta.size());
  for (int i = 0; i < delta.size(); ++i) {
    w[i].resize(a.size());
  }
  for (int j = 0; j < delta.size(); ++j) {
    for (int k = 0; k < a.size(); ++k) {
      w[j][k] = a[k] * delta[j];
    }
  }
  return w;
}

std::vector<double> NeuralNetwork::BP1(std::vector<double>& y) {
  std::vector<double> output(y.size());

  auto d1 = dC_da(a.back(), y);
  auto d2 = dsigma_dz(z.back());
  for (int i = 0; i < d1.size(); ++i) {
    output[i] = d1[i] * d2[i];
  }
  return output;
}

std::vector<double> NeuralNetwork::BP2(std::vector<std::vector<double>>& w, std::vector<double>& delta, std::vector<double>& z) {
  std::vector<double> output(z.size());
  for (int i = 0; i < output.size(); ++i) {
    output[i] = 0;
    for (int j = 0; j < delta.size(); ++j) {
      output[i] += delta[j] * w[j][i];
    }
  }
  Sample s = dsigma_dz(z);
  for (int i = 0; i < s.size(); ++i) {
    output[i] = output[i] * s[i];
  }
  return output;
}

std::vector<std::vector<double>> NeuralNetwork::BP4(std::vector<double>& a, std::vector<double>& delta) {
  return dC_dw(a, delta);
}

std::vector<double> NeuralNetwork::Forward(std::vector<double>& x) {
  a.resize(layers.size());
  for (int i = 0; i < layers.size(); ++i) {
    a[i].resize(layers[i].data().size());
  }
  z.resize(layers.size());
  for (int i = 0; i < layers.size(); ++i) {
    z[i].resize(layers[i].data().size());
  }
  
  for (int i = 0; i < layers.size(); ++i) {
    for (int j = 0; j < layers[i].data().size();++j) {
      auto& v = layers[i].data()[j];
      z[i][j] = 0.0;
      if (i == 0) {
        for (int k = 0; k < x.size(); ++k) {
          z[i][j] += v[k] * x[k];
        }
        a[i][j] = activate(z[i][j]);
      }
      else {
        for (int k = 0; k < a[i - 1].size(); ++k) {
          z[i][j] += v[k] * z[i - 1][k];
        }
        a[i][j] = activate(z[i][j]);
      }
    }
  }
  return a.back();
}

void NeuralNetwork::Backward(std::vector<Sample>& xs, std::vector<double>& y) {
  //auto outputs = Forward(x);
  std::vector<Layer> nLayers(layers.size());
  for (int i = 0; i < nLayers.size(); ++i) {
    nLayers[i].weights.resize(layers[i].weights.size());
    for (int j = 0; j < layers[i].weights[j].size(); ++j) {
      nLayers[i].weights[j].resize(layers[i].weights[j].size(),0);
    }
    nLayers[i].b.resize(layers[i].b.size(), 0);
  }

  for (int i = 0; i < xs.size(); ++i) {
    Forward(xs[i]);

    std::vector<Layer> dlayers = nLayers;
    dlayers[layers.size()-1].b = BP1(y);
    for (int l = layers.size() - 2; l >= 0; --l) {
      dlayers[l].b = BP2(layers[l + 1].data(), dlayers[l + 1].b, z[l]);
      dlayers[l].weights = BP4(a[l - 1], dlayers[l].b);
    }

    for (int l = 0; l < layers.size(); ++l) {
      nLayers[l] = nLayers[l] + dlayers[l];
    }
  }

  //update weights;
  for (int i = 0; i < layers.size(); ++i) {
    nLayers[i] /= xs.size();
    layers[i] = layers[i] + nLayers[i];
  }

}
