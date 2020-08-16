#include <stdio.h>
#include <tchar.h>
#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <tuple>
#include <chrono>

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
	

	printf("\t\texamples:\n");
	printf("\t\t\tVGGNet state\n");
	printf("\t\t\tVGGNet train I:\\CatDog I:\\catdog.pt\n");
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

	VGG_CONFIG config = ctxCmd.enable_batch_norm? VGG_D_BATCHNORM: VGG_D;
	switch (ctxCmd.nn_type)
	{
	case NN_TYPE_VGGA: config = ctxCmd.enable_batch_norm ? VGG_A_BATCHNORM : VGG_A; break;
	case NN_TYPE_VGGA_LRN: config = ctxCmd.enable_batch_norm ? VGG_A_LRN_BATCHNORM : VGG_A_LRN; break;
	case NN_TYPE_VGGB: config = ctxCmd.enable_batch_norm ? VGG_B_BATCHNORM : VGG_B; break;
	case NN_TYPE_VGGC: config = ctxCmd.enable_batch_norm ? VGG_C_BATCHNORM : VGG_C; break;
	case NN_TYPE_VGGD: config = ctxCmd.enable_batch_norm ? VGG_D_BATCHNORM : VGG_D; break;
	case NN_TYPE_VGGE: config = ctxCmd.enable_batch_norm ? VGG_E_BATCHNORM : VGG_E; break;
	}

	VGGNet vgg_net(config, ctxCmd.num_classes);

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to construct the VGG16 network.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
	tm_start = std::chrono::system_clock::now();

	switch (ctxCmd.cmd)
	{
	case NN_CMD_STATE:
		{
			vgg_net.Print();
		}
		break;
	case NN_CMD_TRAIN:
		{
			if (ctxCmd.image_set_root_path.size() == 0 || ctxCmd.train_net_state_path.size() == 0)
			{
				PrintHelp();
				goto done;
			}

			if (ctxCmd.clean_pretrain_net == false && vgg_net.loadnet(ctxCmd.train_net_state_path.c_str()) != 0)
			{
				_tprintf(_T("Failed to load the VGG network from %s, retraining the VGG net.\n"), argv[3]);
			}

			vgg_net.train(ctxCmd.image_set_root_path.c_str(), 
				ctxCmd.train_net_state_path.c_str(), 
				ctxCmd.batchsize,
				ctxCmd.epochnum,
				ctxCmd.learningrate,
				ctxCmd.showloss_per_num_of_batches,
				ctxCmd.clean_pretrain_net);
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
				_tprintf(_T("Failed to load the VGG network from %s.\n"), argv[2]);
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

