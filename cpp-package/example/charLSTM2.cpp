// file: charLSTMDemo.cpp
#pragma warning(disable: 4996)  // VS2015 complains on 'std::copy' ...
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"
#include "utils.h"

using namespace mxnet::cpp;
bool TIME_MAJOR = true;

namespace rnn_enum {
    enum InputFormat {TNC, NTC};
}

Executor* LSTMExecutor(Symbol* lstm, std::map<std::string, NDArray>* args_map) {
  Context device(DeviceType::kCPU, 0);
  lstm->InferArgsMap(device, args_map, *args_map);
  return lstm->SimpleBind(device, *args_map);
}

int main() {
  const int sequence_length = 100; 
  const int batch_size = 8; 
  const int num_layer = 2;
  const int input_dim = 768;
  const int state_dim = 1024;

  const int directions = 1;
  const int ngates = 4;
  const int input_format = rnn_enum::TNC;
  const mx_float dropout = 0.0;
  Context device(DeviceType::kCPU, 0);

  auto data = Symbol::Variable("data");

  // We need not do the SwapAxis op as python version does. Direct and better performance in C++!
  auto rnn_h_init = Symbol::Variable("LSTM_init_h");
  auto rnn_c_init = Symbol::Variable("LSTM_init_c");
  auto rnn_params = Symbol::Variable("LSTM_parameters");  // See explanations near RNNXavier class
  auto variable_sequence_length = Symbol::Variable("sequence_length");
  auto rnn = RNN(data, rnn_params, rnn_h_init, rnn_c_init, variable_sequence_length, state_dim,
                 num_layer, RNNMode::kLstm, false, dropout, true);

  for (auto c : rnn.ListArguments()) std::cout << c << std::endl;
  auto out = Symbol::Group({ rnn[1/*RNNOpOutputs::kStateOut=1*/],
    rnn[2/*RNNOpOutputs::kStateCellOut=2*/] });

  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(sequence_length, batch_size, input_dim), device, false);
  // Avoiding SwapAxis, batch_size is of second dimension.
  args_map["LSTM_init_c"] = NDArray(Shape(num_layer, batch_size, state_dim), device, false);
  args_map["LSTM_init_h"] = NDArray(Shape(num_layer, batch_size, state_dim), device, false);
  std::vector<mx_float> zeros(batch_size * 1 * num_layer * state_dim, 0);
  args_map["LSTM_init_c"].SyncCopyFromCPU(zeros);
  args_map["LSTM_init_h"].SyncCopyFromCPU(zeros);
  std::vector<mx_float> src(sequence_length * batch_size * input_dim, 0);
  args_map["data"].SyncCopyFromCPU(src);
  NDArray::WaitAll();
  std::cout << "Argument size: " << out.ListArguments().size() << std::endl;
  for (auto c : out.ListArguments()) std::cout << c << std::endl;
  
  out.InferArgsMap(device, &args_map, args_map);
  Executor* exe = out.SimpleBind(device, args_map);
  exe->Forward(false);

  delete exe;
  return 0;
}