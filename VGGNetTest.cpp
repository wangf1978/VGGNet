#include <stdio.h>
#include <tchar.h>
#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <tuple>
#include <chrono>
#include <io.h>

#define NOMINMAX

#include <wincodec.h>
#include <wincodecsdk.h>
#include <wrl/client.h>
#include <d3d.h>
#include <d2d1.h>
#include <d2d1_2.h>
#include <shlwapi.h>

#include "VGGNet.h"
#include "ImageProcess.h"
#include "CmdLineParser.h"
#include "BaseNNet.h"

extern std::map<VGG_CONFIG, std::string> _VGG_CONFIG_NAMES;

void FreeBlob(void* p)
{
	//printf("Free the blob which is loaded by a Tensor.\n");
	free(p);
}

#define NUM_OF_CLASSES		1000

void PrintHelp()
{
	printf("Usage:\n\tVGGNet [options] command [args...]\n");

	printf("\t\tcommands:\n");
	printf("\t\t\tstate:\t\tPrint the VGG layout\n");
	printf("\t\t\ttrain:\t\tTrain the VGG16\n");
	printf("\t\t\tverify:\t\tVerify the train network with the test set\n");
	printf("\t\t\tclassify:\tClassify the input image\n");

	printf("\t\targs:\n");
	printf("\t\t\t--batchsize, -b\tThe batch size of training the network\n");
	printf("\t\t\t--epochnum\tSpecify how many train epochs the network will be trained for\n");
	printf("\t\t\t--lr, -l\tSpecify the learning rate\n");
	printf("\t\t\t--batchnorm,\n\t\t\t--bn\t\tEnable batchnorm or not\n");
	printf("\t\t\t--numclass\tSpecify the num of classes of output\n");
	printf("\t\t\t--smallsize, -s\tUse 32x32 input image or not\n");
	printf("\t\t\t--showloss, -s\tSpecify how many batches the loss rate is print once\n");
	printf("\t\t\t--clean\t\tclean the previous train result\n");

	printf("\t\texamples:\n");
	printf("\t\t\tVGGNet state\n");
	printf("\t\t\tVGGNet train I:\\CatDog I:\\catdog.pt --bn -b 64 --showloss 10 --lr 0.001\n");
	printf("\t\t\tVGGNet verify I:\\CatDog I:\\catdog.pt\n");
	printf("\t\t\tVGGNet classify I:\\catdog.pt I:\\test.png\n");
}

void freeargv(int argc, char** argv)
{
	if (argv == NULL)
		return;

	for (int i = 0; i < argc; i++)
	{
		if (argv[i] == NULL)
			continue;

		delete[] argv[i];
	}
	delete argv;
}

void Test()
{
	ImageProcess imageprocessor;
	if (SUCCEEDED(imageprocessor.Init(10, 10)))
	{
		torch::Tensor tensor;
		{
			float means[3] = { 0.f, 0.f, 0.f };
			float stds[3] = { 1.f, 1.f, 1.f };
			imageprocessor.SetRGBMeansAndStds(means, stds);
		}
		if (SUCCEEDED(imageprocessor.ToTensor(_T("I:\\RGB.png"), tensor)))
		{
			printf("before transforming....\n");
			std::cout << tensor << '\n';
		}

		{
			float means[3] = { 0.5f, 0.5f, 0.5f };
			float stds[3] = { 0.5f, 0.5f, 0.5f };
			imageprocessor.SetRGBMeansAndStds(means, stds);
		}
		if (SUCCEEDED(imageprocessor.ToTensor(_T("I:\\RGB.png"), tensor)))
		{
			printf("after transforming....\n");
			std::cout << tensor << '\n';
		}
	}

	imageprocessor.Uninit();
}

int _tmain(int argc, const TCHAR* argv[])
{
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	if (argc <= 1)
	{
		PrintHelp();
		return 0;
	}

	const char** u8argv = NULL;
#ifdef _UNICODE
	u8argv = new const char*[argc];
	for (int i = 0; i < argc; i++)
	{
		if (argv[i] == NULL)
			u8argv[i] = NULL;
		else
		{
			size_t ccLen = _tcslen(argv[i]);
			u8argv[i] = new const char[ccLen * 4 + 1];
			WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, (LPSTR)u8argv[i], (int)ccLen * 4 + 1, NULL, NULL);
		}
	}
#else
	u8argv = (const char**)argv;
#endif

	if (CmdLineParser::ProcessCommandLineArgs(argc, u8argv) == false)
	{
		freeargv(argc, (char**)u8argv);
		PrintHelp();
		return -1;
	}

	//CmdLineParser::GetCmdLineParser().Print();

	if (FAILED(CoInitializeEx(NULL, COINIT_MULTITHREADED)))
	{
		freeargv(argc, (char**)u8argv);
		return -1;
	}

	CmdLineParser& ctxCmd = CmdLineParser::GetCmdLineParser();
	VGG_CONFIG config = ctxCmd.enable_batch_norm ? VGG_D_BATCHNORM : VGG_D;
	switch (ctxCmd.nn_type)
	{
	case NN_TYPE_VGGA: config = ctxCmd.enable_batch_norm ? VGG_A_BATCHNORM : VGG_A; break;
	case NN_TYPE_VGGA_LRN: config = ctxCmd.enable_batch_norm ? VGG_A_LRN_BATCHNORM : VGG_A_LRN; break;
	case NN_TYPE_VGGB: config = ctxCmd.enable_batch_norm ? VGG_B_BATCHNORM : VGG_B; break;
	case NN_TYPE_VGGC: config = ctxCmd.enable_batch_norm ? VGG_C_BATCHNORM : VGG_C; break;
	case NN_TYPE_VGGD: config = ctxCmd.enable_batch_norm ? VGG_D_BATCHNORM : VGG_D; break;
	case NN_TYPE_VGGE: config = ctxCmd.enable_batch_norm ? VGG_E_BATCHNORM : VGG_E; break;
	}

	VGGNet vgg_net;
	switch (ctxCmd.cmd)
	{
	case NN_CMD_STATE:
		{
			if (_access(ctxCmd.train_net_state_path.c_str(), 0) == 0 && vgg_net.loadnet(ctxCmd.train_net_state_path.c_str()) == 0)
				vgg_net.Print();
			else if (vgg_net.loadnet(config, ctxCmd.num_classes, ctxCmd.use_32x32_input) == 0)
				vgg_net.Print();
		}
		break;
	case NN_CMD_TRAIN:
		{
			bool bLoadSucc = false;
			if (ctxCmd.image_set_root_path.size() == 0 || ctxCmd.train_net_state_path.size() == 0)
			{
				PrintHelp();
				goto done;
			}

			// delete the previous pre-train net state
			if (ctxCmd.clean_pretrain_net)
			{
				if (DeleteFileA(ctxCmd.train_net_state_path.c_str()) == FALSE)
				{
					printf("Failed to delete the file '%s' {err: %lu}.\n", ctxCmd.train_net_state_path.c_str(), GetLastError());
				}
			}

			// Try to load the net from the pre-trained file if it exist
			if (_access(ctxCmd.train_net_state_path.c_str(), 0) == 0)
			{
				if (vgg_net.loadnet(ctxCmd.train_net_state_path.c_str()) != 0)
					printf("Failed to load the VGG network from %s, retraining the VGG net.\n", ctxCmd.train_net_state_path.c_str());
				else
				{
					// Check the previous neutral network config is the same with current specified parameters
					if (config != vgg_net.getcurrconfig() ||
						ctxCmd.num_classes != vgg_net.getnumclasses() ||
						ctxCmd.use_32x32_input != vgg_net.isuse32x32input())
					{
						printf("The pre-trained network config is different with the specified parameters:\n");
						auto iter1 = _VGG_CONFIG_NAMES.find(vgg_net.getcurrconfig());
						auto iter2 = _VGG_CONFIG_NAMES.find(config);
						printf("\tcurrent config: %s, the specified config: %s\n",
							iter1 != _VGG_CONFIG_NAMES.end() ? iter1->second.c_str() : "Unknown",
							iter2 != _VGG_CONFIG_NAMES.end() ? iter2->second.c_str() : "Unknown");
						printf("\tcurrent numclass: %d, the specified numclass: %d\n", vgg_net.getnumclasses(), ctxCmd.num_classes);
						printf("\tcurrent use_32x32_input: %s, the specified use_32x32_input: %s\n", 
							vgg_net.isuse32x32input()?"yes":"no", ctxCmd.use_32x32_input?"yes":"no");
						printf("Continue using the network config in the pre-train net state...\n");
					}

					bLoadSucc = true;
				}
			}

			// Failed to load net from the previous trained net, retrain the net
			if (bLoadSucc == false && vgg_net.loadnet(config, ctxCmd.num_classes, ctxCmd.use_32x32_input) != 0)
			{
				printf("Failed to load the neutral network.\n");
				goto done;
			}

			tm_end = std::chrono::system_clock::now();

			{
				auto iter_config = _VGG_CONFIG_NAMES.find(vgg_net.getcurrconfig());
				long long load_duration =
					std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
				printf("It took %lldh:%02dm:%02d.%03ds msec to construct the '%s' network.\n",
					load_duration / 1000 / 3600,
					load_duration / 1000 / 60 % 60,
					load_duration / 1000 % 60,
					load_duration % 1000,
					iter_config != _VGG_CONFIG_NAMES.end() ? iter_config->second.c_str() : "Unknown");
				tm_start = std::chrono::system_clock::now();
			}

			vgg_net.train(ctxCmd.image_set_root_path.c_str(), 
				ctxCmd.train_net_state_path.c_str(), 
				ctxCmd.batchsize,
				ctxCmd.epochnum,
				ctxCmd.learningrate,
				ctxCmd.showloss_per_num_of_batches);
		}
		break;
	case NN_CMD_VERIFY:
		{
			if (ctxCmd.image_set_root_path.size() == 0 || ctxCmd.train_net_state_path.size() == 0)
			{
				PrintHelp();
				goto done;
			}

			vgg_net.verify(ctxCmd.image_set_root_path.c_str(), ctxCmd.train_net_state_path.c_str());
		}
		break;
	case NN_CMD_CLASSIFY:
		{
			if (ctxCmd.train_net_state_path.size() == 0 || ctxCmd.image_path.size() == 0)
			{
				PrintHelp();
				goto done;
			}

			if (vgg_net.loadnet(ctxCmd.train_net_state_path.c_str()) != 0)
			{
				_tprintf(_T("Failed to load the VGG network from %s.\n"), ctxCmd.train_net_state_path.c_str());
				goto done;
			}

			vgg_net.classify(ctxCmd.image_path.c_str());
		}
		break;
	case NN_CMD_TEST:
		Test();
		break;
	default:
		{
			PrintHelp();
			goto done;
		}
	}

done:
	CoUninitialize();

	freeargv(argc, (char**)u8argv);

	return 0;
}

