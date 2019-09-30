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
  Context device(DeviceType::kCPU, 0);

  const int first_layer_size = (input_dim * state_dim + state_dim * state_dim + state_dim * 2) * ngates;
  const int rest_layer_size = (state_dim * directions * state_dim + state_dim * state_dim + state_dim * 2)
      * ngates * (num_layer - 1);
  const int param_size = (first_layer_size + rest_layer_size) * directions;

  const float mean = 0.0;
  const float sigma = 1.0;
  std::default_random_engine generator(42);
  std::normal_distribution<float> distribution(mean, sigma);

  const float dropout = 0.0;

  auto data_ = Symbol::Variable("data");
  if (input_format == rnn_enum::NTC)
    data_ = SwapAxis(data_, 0, 1);

  auto rnn_h_init = Symbol::Variable("LSTM_init_h");
  auto rnn_c_init = Symbol::Variable("LSTM_init_c");
  auto rnn_weights = Symbol::Variable("LSTM_parameters");
  auto variable_sequence_length = Symbol::Variable("sequence_length");
  auto rnn = RNN("LSTM", data_, rnn_weights, rnn_h_init, rnn_c_init, variable_sequence_length, state_dim,
                 num_layer, RNNMode::kLstm, false, dropout, true);

  auto lstm = Symbol::Group({ rnn[0/*RNNOuputs::kOut=0*/], rnn[1/*RNNOpOutputs::kStateOut=1*/],
    rnn[2/*RNNOpOutputs::kStateCellOut=2*/] });
  for (auto c : rnn[0].ListArguments()) std::cout << c << std::endl;

  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;
  lstm.InferExecutorArrays(device, &arg_arrays, &grad_arrays, &grad_reqs,
                           &aux_arrays, args_map, std::map<std::string, NDArray>(),
                           std::map<std::string, OpReqType>(), aux_map);
  std::cout << "args_map size: " << args_map.size() << std::endl;
  std::cout << "aux_map size: " << aux_map.size() << std::endl;

  std::vector<mx_float> parameter(param_size, 0);
  std::generate(parameter.begin(), parameter.end(),
      [&distribution, &generator](){ return distribution(generator); });
  
  args_map["LSTM_parameters"] = NDArray(Shape(param_size), device);
  args_map["LSTM_parameters"].SyncCopyFromCPU(parameter);
  args_map["LSTM_parameters"].WaitToRead();

  args_map["sequence_length"] = NDArray(Shape(batch_size), device, false, 4);

  args_map["data"] = NDArray(Shape(sequence_length, batch_size, input_dim), device);
  args_map["LSTM_init_h"] = NDArray(Shape(num_layer * 1, batch_size, state_dim), device);
  args_map["LSTM_init_c"] = NDArray(Shape(num_layer * 1, batch_size, state_dim), device);
  
  std::vector<mx_float> zeros(num_layer * 1 * batch_size * state_dim, 0);
  args_map["LSTM_init_h"].SyncCopyFromCPU(zeros);
  args_map["LSTM_init_c"].SyncCopyFromCPU(zeros);
  args_map["LSTM_init_h"].WaitToRead();
  args_map["LSTM_init_c"].WaitToRead();

  auto exe = LSTMExecutor(&lstm, &args_map);

  std::vector<mx_float> data(sequence_length * batch_size * input_dim);

  std::vector<float> output(sequence_length * batch_size * state_dim);
  std::vector<float> stateoutput(num_layer * 1 * batch_size * state_dim);
  std::vector<float> statecelloutput(num_layer * 1 * batch_size * state_dim);

  const int warmup = 10;
  const int run = 50;
  int64_t milliseconds = 0;
  for (size_t iter = 0; iter < warmup + run; ++iter) {
    std::generate(data.begin(), data.end(),
        [&distribution, &generator](){ return distribution(generator); });
    
    auto begin = std::chrono::high_resolution_clock::now();
    exe->arg_dict()["data"].SyncCopyFromCPU(data);
    NDArray::WaitAll();
    exe->Forward(false);
    
    exe->outputs[0].SyncCopyToCPU(output.data(), output.size());
    exe->outputs[1].SyncCopyToCPU(stateoutput.data(), stateoutput.size());
    exe->outputs[2].SyncCopyToCPU(statecelloutput.data(), statecelloutput.size());
    NDArray::WaitAll();

    auto end = std::chrono::high_resolution_clock::now();
    auto period = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    if (iter > warmup) milliseconds += period.count();
  }
  std::cout << "Throughput: " << batch_size * run * 1.0 / milliseconds << " sample/ms";

  delete exe;
  return 0;
}