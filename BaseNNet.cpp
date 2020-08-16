#include "BaseNNet.h"
#include "util.h"
#include <assert.h>
#include <memory>

int BaseNNet::LoadNet(const char* szNNName)
{
	int iRet = 0;
	if (xmlDoc.LoadFile("nnconfig.xml") != tinyxml2::XML_SUCCESS)
	{
		printf("Failed to load 'nnconfig.xml'.\n");
		return -1;
	}

	if (xmlDoc.RootElement() == NULL || xmlDoc.RootElement()->NoChildren() || XP_STRICMP(xmlDoc.RootElement()->Name(), "nns") != 0)
	{
		printf("It is an invalid 'nnconfig.xml' file.\n");
		return -1;
	}

	auto child = xmlDoc.RootElement()->FirstChildElement("nn");
	while (child != NULL)
	{
		if (XP_STRICMP(child->Attribute("name"), szNNName) == 0)
			break;
		child = child->NextSiblingElement("nn");
	}

	if (child == NULL)
	{
		printf("Failed to find the neutral network '%s'.\n", szNNName);
		return -1;
	}

	// Load the modules one by one
	auto Elementmodules = child->FirstChildElement("modules");
	if (Elementmodules == NULL || Elementmodules->NoChildren())
	{
		printf("No modules to be loaded.\n");
		return -1;
	}

	auto ElementModule = Elementmodules->FirstChildElement("module");
	while (ElementModule != NULL)
	{
		if (LoadModule(ElementModule) != 0)
		{
			iRet = -1;
			printf("Failed to load the module '%s'.\n", ElementModule->Attribute("name", "(null)"));
			goto done;
		}
		ElementModule = ElementModule->NextSiblingElement("module");
	}

	// load the forward list
	auto ElementForward = child->FirstChildElement("forward");
	if (ElementForward != NULL && !ElementForward->NoChildren())
	{
		auto f = ElementForward->FirstChildElement("f");
		while (f != NULL)
		{
			forward_list.push_back(f);
			f = f->NextSiblingElement();
		}
	}

	nn_model_name = szNNName;

	iRet = 0;

done:
	return iRet;
}

int BaseNNet::UnloadNet()
{
	for (auto& iter = nn_modules.begin(); iter != nn_modules.end(); iter++)
	{
		unregister_module(iter->first);
		iter->second = nullptr;
	}

	nn_modules.clear();
	forward_list.clear();

	return 0;
}

void BaseNNet::Print()
{
	std::cout << "Neutral Network '" << nn_model_name << "' layout:\n";

	std::cout << *this << '\n';

	// print the total numbers of parameters and weight layers
	int64_t total_parameters = 0, weight_layers = 0;
	auto netparams = parameters();
	for (size_t i = 0; i < netparams.size(); i++)
	{
		int64_t param_item_count = netparams[i].sizes().size() > 0 ? 1 : 0;
		for (size_t j = 0; j < netparams[i].sizes().size(); j++)
			param_item_count *= netparams[i].size(j);
		total_parameters += param_item_count;
	}

	std::cout << "\nparameters layout:\n\ttotal parameters: " << total_parameters << ", weight layers: " << netparams.size() / 2 << '\n';

	// print each parameter layout
	for (auto const& p : named_parameters())
		std::cout << "\t" << p.key() << ":\n\t\t" << p.value().sizes() << '\n';

	return;
}

torch::Tensor BaseNNet::forward(torch::Tensor& input)
{
	namespace F = torch::nn::functional;

	if (forward_list.size() == 0)
	{

	}
	else
	{
		for (auto& f : forward_list)
		{
			const char* szModuleName = f->Attribute("module");
			if (f != NULL)
			{
				auto m = nn_modules.find(szModuleName);
				if (m != nn_modules.end())
				{
					std::string& module_type = nn_module_types[szModuleName];
					if (module_type == "conv2d")
					{
						auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m->second);
						input = spConv2d->forward(input);
						continue;
					}
					else if (module_type == "linear")
					{
						auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m->second);
						input = spLinear->forward(input);
						continue;
					}
					else if (module_type == "batchnorm2d")
					{
						auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m->second);
						input = spBatchNorm2d->forward(input);
						continue;
					}
				}
			}

			const char* szFunctional = f->Attribute("functional");
			if (szFunctional != NULL)
			{
				if (XP_STRICMP(szFunctional, "relu") == 0)
				{
					bool inplace = f->BoolAttribute("inplace", false);
					input = F::relu(input, F::ReLUFuncOptions(inplace));
					continue;
				}
				else if (XP_STRICMP(szFunctional, "max_pool2d") == 0)
				{
					int64_t kernel_size = f->Int64Attribute("kernel_size", 2);
					input = F::max_pool2d(input, F::MaxPool2dFuncOptions(kernel_size));
					continue;
				}
				else if (XP_STRICMP(szFunctional, "dropout") == 0)
				{
					double p = f->DoubleAttribute("p", 0.5);
					bool inplace = f->BoolAttribute("inplace", false);
					input = F::dropout(input, F::DropoutFuncOptions().p(p).inplace(inplace));
					continue;
				}
			}

			const char* szView = f->Attribute("view");
			if (szView != NULL)
			{
				if (XP_STRICMP(szView, "flat"))
				{
					input = input.view({ input.size(0), -1 });
					continue;
				}
			}
		}
	}

	return input;
}

int BaseNNet::LoadModule(tinyxml2::XMLElement* moduleElement)
{
	int iRet = 0;
	if (moduleElement == NULL)
		return -1;

	const char* szModuleType = moduleElement->Attribute("type");
	if (szModuleType == NULL)
	{
		printf("Please specify the module type.\n");
		return -1;
	}

	const char* szModuleName = moduleElement->Attribute("name");
	if (szModuleName == NULL)
	{
		printf("Please specify the module name.\n");
		return -1;
	}

	if (XP_STRICMP(szModuleType, "conv2d") == 0)
	{
		// extract the attributes of conv2d
		int64_t in_channels = moduleElement->Int64Attribute("in_channels", -1LL); assert(in_channels > 0);
		int64_t out_channels = moduleElement->Int64Attribute("out_channels", -1LL); assert(out_channels > 0);
		int64_t kernel_size = moduleElement->Int64Attribute("kernel_size", -1LL); assert(kernel_size > 0);
		int64_t padding = moduleElement->Int64Attribute("padding", 1LL);

		std::shared_ptr<torch::nn::Module> spConv2D = 
			std::make_shared<torch::nn::Conv2dImpl>(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(padding));
		nn_modules[szModuleName] = spConv2D;
		nn_module_types[szModuleName] = szModuleType;
		register_module(szModuleName, spConv2D);
	}
	else if (XP_STRICMP(szModuleType, "linear") == 0)
	{
		int64_t in_features = moduleElement->Int64Attribute("in_features", -1LL); assert(in_features > 0);
		int64_t out_features = moduleElement->Int64Attribute("out_features", -1LL); assert(out_features > 0);

		std::shared_ptr<torch::nn::Module> spLinear =
			std::make_shared<torch::nn::LinearImpl>(in_features, out_features);
		nn_modules[szModuleName] = spLinear;
		register_module(szModuleName, spLinear);
	}
	else if (XP_STRICMP(szModuleType, "batchnorm2d") == 0)
	{
		int64_t num_features = moduleElement->Int64Attribute("num_features", -1); assert(num_features > 0);

		std::shared_ptr<torch::nn::Module> spBatchNorm2D =
			std::make_shared<torch::nn::BatchNorm2dImpl>(num_features);
		nn_modules[szModuleName] = spBatchNorm2D;
		register_module(szModuleName, spBatchNorm2D);
	}
	else
	{
		printf("Failed to load the module with unsupported type: '%s'.\n", szModuleType);
		iRet = -1;
	}

	return iRet;
}