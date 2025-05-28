#pragma once
#include "Layer.h"
#include "Activation.h"
#include <memory>

using Sample = std::vector<double>;

/*
Formula of BP
delta = (dC/da) h \sigma'


*/

class NeuralNetwork
{
public:
  NeuralNetwork(double lr = 1.0)
    :learningRate(lr){
    activation = std::make_shared<RELU>();
  }

  void SetLearningRate(double lr) {
    learningRate = lr;
  }

  void AddLayer(Layer& layer) {
    layers.push_back(layer);
  }

  void SetActivation(std::shared_ptr<Activation> active) {
    activation = active;
  }

  std::vector<double> Forward(std::vector<double>& x);

  void Backward(std::vector<Sample>& x, std::vector<Sample>& y);

  double ComputeLoss(std::vector<Sample>& xs, std::vector<Sample>& ys);

  void Train(std::vector<Sample>& xs, std::vector<Sample>& ys, int maxEpochs, double tolerance);

  void Print();

protected:

  std::vector<double> dC_da(std::vector<double>& a, std::vector<double>& y);
  std::vector<double> dsigma_dz(std::vector<double>& z);
  //std::vector<double> dC_db(std::vector<double>& b);
  std::vector<std::vector<double>> dC_dw(Sample& a, Sample& delta);

  std::vector<double> BP1(std::vector<double>& input);
  std::vector<double> BP2(std::vector<std::vector<double>>& w, std::vector<double>& delta, std::vector<double>& z);
  //std::vector<double> BP3()
  std::vector<std::vector<double>> BP4(std::vector<double>& a, std::vector<double>& delta);



protected:
  std::vector<Layer> layers;
  std::shared_ptr<Activation> activation;
  double learningRate = 1.0;

  //temp
  std::vector<std::vector<double>> a;
  std::vector<std::vector<double>> z;
};

