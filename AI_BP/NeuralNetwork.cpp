#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <string>

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
    b[i] = activation->dActivate(z[i]);
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
    a[i].resize(layers[i].data().size(),0);
  }
  z.resize(layers.size());
  for (int i = 0; i < layers.size(); ++i) {
    z[i].resize(layers[i].data().size(),0);
  }
  
  for (int i = 0; i < layers.size(); ++i) {
    for (int j = 0; j < layers[i].data().size();++j) {
      auto& v = layers[i].data()[j];
      z[i][j] = 0.0;
      if (i == 0) {
        for (int k = 0; k < x.size(); ++k) {
          z[i][j] += v[k] * x[k];
        }
        z[i][j] += layers[i].b[j];
        a[i][j] = activation->activate(z[i][j]);
      }
      else {
        for (int k = 0; k < a[i - 1].size(); ++k) {
          z[i][j] += v[k] * a[i - 1][k];
        }
        z[i][j] += layers[i].b[j];
        a[i][j] = activation->activate(z[i][j]);
      }
    }
  }
  return a.back();
}

void NeuralNetwork::Backward(std::vector<Sample>& xs, std::vector<Sample>& ys) {
  std::vector<Layer> nLayers(layers.size());
  for (int i = 0; i < nLayers.size(); ++i) {
    nLayers[i].weights.resize(layers[i].weights.size());
    for (int j = 0; j < layers[i].weights.size(); ++j) {
      nLayers[i].weights[j].resize(layers[i].weights[j].size(),0);
    }
    nLayers[i].b.resize(layers[i].b.size(), 0);
  }
  std::vector<Layer> resultLayers = nLayers;

  for (int i = 0; i < xs.size(); ++i) {
    Forward(xs[i]);

    std::vector<Layer> dlayers = nLayers;
    dlayers[layers.size()-1].b = BP1(ys[i]);
    for (int l = layers.size() - 2; l >= 0; --l) {
      dlayers[l].b = BP2(layers[l + 1].data(), dlayers[l + 1].b, z[l]);
      if (l >= 1) {
        dlayers[l].weights = BP4(a[l - 1], dlayers[l].b);
      }
      else {
        dlayers[l].weights = BP4(xs[i], dlayers[l].b);
      }
    }

    for (int l = 0; l < layers.size(); ++l) {
      resultLayers[l] = resultLayers[l] + dlayers[l];
    }
  }

  //update weights;
  for (int i = 0; i < resultLayers.size(); ++i) {
    resultLayers[i] /= xs.size();
    layers[i].ApplyGradient(resultLayers[i],learningRate);
  }

}

double NeuralNetwork::ComputeLoss(std::vector<Sample>& xs, std::vector<Sample>& ys) {
  double totalLoss = 0.0;
  for (int i = 0; i < xs.size(); ++i) {
    std::vector<double> pred = Forward(xs[i]);
    for (int j = 0; j < pred.size(); ++j) {
      double diff = pred[j] - ys[i][j];
      totalLoss += diff * diff;
    }
  }
  return totalLoss / xs.size();
}

void NeuralNetwork::Train(std::vector<Sample>& xs, std::vector<Sample>& ys,
  int maxEpochs, double tolerance) {
  for (int epoch = 0; epoch < maxEpochs; ++epoch) {
    Backward(xs, ys);
    double loss = ComputeLoss(xs, ys);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;

    if (loss < tolerance) {
      std::cout << "Converged at epoch " << epoch << std::endl;
      break;
    }
  }
}



void NeuralNetwork::Print() {
  for (int i = 0; i < layers.size(); ++i) {
    std::cout << "layer:" << i << std::endl;
    auto& data = layers[i].data();
    for (int j = 0; j < data.size(); ++j) {
      for (int k = 0; k < data[j].size(); ++k) {
        std::cout << "W_" << k << "," << j << "=" << data[j][k] <<" ";
      }
      std::cout << std::endl;
    }
    for (int j = 0; j < layers[i].b.size(); ++j) {
      std::cout << "B_" << j << "=" << layers[i].b[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }
}
