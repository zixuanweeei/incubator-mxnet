// file: charLSTMDemo.cpp
#pragma warning(disable: 4996)  // VS2015 complains on 'std::copy' ...
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"
#include "utils.h"

using namespace mxnet::cpp;
bool TIME_MAJOR = true;

int main() {
  const int batch_size = 8;
  const int image_size = 26;

  Context ctx = Context::cpu();
  auto x = Symbol::Variable("X");
  auto label = Symbol::Variable("label");

  Symbol weight = Symbol::Variable("Weight");
  Symbol bias = Symbol::Variable("Bias");
  Symbol output = Symbol::Variable("Output");

  Symbol fc = FullyConnected(x, weight, bias, 768);

  std::map<std::string, NDArray> args;
  args["X"] = NDArray(Shape(batch_size, image_size*image_size), ctx);
  args["label"] = NDArray(Shape(batch_size), ctx);
  // Let MXNet infer shapes other parameters such as weights
  fc.InferArgsMap(ctx, &args, args);

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  auto initializer = Uniform(0.01);
  for (auto& arg : args) {
    // arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }
  std::cout << "FC arguments size: " << fc.ListArguments().size() << std::endl;

  auto *exec = fc.SimpleBind(ctx, args);
  std::vector<mx_float> src(batch_size * image_size * image_size, 0);
  args["X"].SyncCopyFromCPU(src);
  exec->Forward(false);
  auto array_out = exec->outputs[0];
  array_out.WaitToRead();

  delete exec;
  return 0;
}