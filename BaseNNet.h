#pragma once

#include <torch/torch.h>
#include "tinyxml2.h"
#include <map>
#include <vector>

using namespace torch::nn;

using NNetOptions = torch::OrderedDict<std::string, std::string>;

class BaseNNet: public Module
{
public:
	int SetOptions(NNetOptions options);
	int Init(const char* szNNName);
	int Uninit();

	void Print();

	torch::Tensor forward(torch::Tensor& input);

protected:
	int LoadModule(tinyxml2::XMLElement* moduleElement);

private:
	std::map<std::string, std::shared_ptr<torch::nn::Module>> nn_modules;
	std::map<std::string, std::string > nn_module_types;
	std::vector<tinyxml2::XMLElement*> forward_list;
	tinyxml2::XMLDocument xmlDoc;
	std::string nn_model_name;
	NNetOptions nn_options;
};

