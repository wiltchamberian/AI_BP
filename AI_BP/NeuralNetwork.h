#pragma once
#include "Layer.h"

using Sample = std::vector<double>;

/*
Formula of BP
delta = (dC/da) h \sigma'


*/

class NeuralNetwork
{
public:
  void AddLayer(Layer& layer) {
    layers.push_back(layer);
  }

  std::vector<double> dC_da(std::vector<double>& a, std::vector<double>& y);
  std::vector<double> dsigma_dz(std::vector<double>& z);
  //std::vector<double> dC_db(std::vector<double>& b);
  std::vector<std::vector<double>> dC_dw(Sample& a, Sample& delta);

  std::vector<double> BP1(std::vector<double>& input);
  std::vector<double> BP2(std::vector<std::vector<double>>& w, std::vector<double>& delta, std::vector<double>& z);
  //std::vector<double> BP3()
  std::vector<std::vector<double>> BP4(std::vector<double>& a, std::vector<double>& delta);
  
  //RELU
  double activate(double x);
  double dActivate(double x);

  std::vector<double> Forward(std::vector<double>& x);

  void Backward(std::vector<Sample>& x, std::vector<double>& y);

protected:
  std::vector<Layer> layers;

  //temp
  std::vector<std::vector<double>> a;
  std::vector<std::vector<double>> z;
};

