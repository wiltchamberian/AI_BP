#include <iostream>
#include "NeuralNetwork.h"

int main()
{
  NeuralNetwork network;
  Layer layer(10, 4);
  network.AddLayer(layer);
}

