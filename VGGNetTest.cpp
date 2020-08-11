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

void FreeBlob(void* p)
{
	//printf("Free the blob which is loaded by a Tensor.\n");
	free(p);
}

#define NUM_OF_CLASSES		2

void PrintHelp()
{
	printf("Usage:\n\tVGGNet [command] [train/test image set] [train_net_state_filename] [testimagefile]\n");

	printf("\t\tcommands:\n");
	printf("\t\t\tstate:\t\tPrint the VGG layout\n");
	printf("\t\t\ttrain:\t\tTrain the VGG16\n");
	printf("\t\t\tverify:\t\tVerify the train network with the test set\n");
	printf("\t\t\tclassify:\tClassify the input image\n");

	printf("\t\texamples:\n");
	printf("\t\t\tVGGNet state\n");
	printf("\t\t\tVGGNet train I:\\CatDog I:\\catdog.pt\n");
	printf("\t\t\tVGGNet verify I:\\CatDog I:\\catdog.pt\n");
	printf("\t\t\tVGGNet classify I:\\catdog.pt I:\\test.png\n");
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

	if (FAILED(CoInitializeEx(NULL, COINIT_MULTITHREADED)))
		return -1;

	VGGNet vgg16_net(NUM_OF_CLASSES);

	// Test image processor, and convert the image to torch tensor
#if 0
	ImageProcess imageprocessor;
	if (SUCCEEDED(imageprocessor.Init(10, 10)))
	{
		torch::Tensor tensor;
		if (SUCCEEDED(imageprocessor.ToTensor(_T("I:\\RGB.png"), tensor)))
		{
			printf("before transforming....\n");
			std::cout << tensor << '\n';
		}

		if (SUCCEEDED(imageprocessor.ToTensor(_T("I:\\RGB.png"), tensor, 0.5f, 0.5f)))
		{
			printf("after transforming....\n");
			std::cout << tensor << '\n';
		}
	}

	imageprocessor.Uninit();
#endif

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to construct the VGG16 network.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
	tm_start = std::chrono::system_clock::now();

	if (_tcsicmp(argv[1], _T("state")) == 0)
	{
		// print the VGG16 layout
		std::cout << "VGG16 layout:\n" << vgg16_net << '\n';

		// print the total numbers of parameters and weight layers
		int64_t total_parameters = 0, weight_layers = 0;
		auto netparams = vgg16_net.parameters();
		for (size_t i = 0; i < netparams.size(); i++)
		{
			int64_t param_item_count = netparams[i].sizes().size() > 0 ? 1 : 0;
			for (size_t j = 0; j < netparams[i].sizes().size(); j++)
				param_item_count *= netparams[i].size(j);
			total_parameters += param_item_count;
		}

		std::cout << "\nparameters layout:\n\ttotal parameters: " << total_parameters << ", weight layers: " << netparams.size() / 2 << '\n';

		// print each parameter layout
		for (auto const& p : vgg16_net.named_parameters())
			std::cout << "\t" << p.key() << ":\n\t\t" << p.value().sizes() << '\n';
	}
	else if (_tcsicmp(argv[1], _T("train")) == 0)
	{
		if (argc <= 3)
		{
			PrintHelp();
			goto done;
		}

		if (vgg16_net.loadnet(argv[3]) != 0)
		{
			_tprintf(_T("Failed to load the VGG network from %s, retraining the VGG net.\n"), argv[3]);
		}

		vgg16_net.train(argv[2], argv[3]);
	}
	else if (_tcsicmp(argv[1], _T("verify")) == 0)
	{
		if (argc <= 3)
		{
			PrintHelp();
			goto done;
		}

		vgg16_net.verify(argv[2], argv[3]);
	}
	else if (_tcsicmp(argv[1], _T("classify")) == 0)
	{
		if (argc <= 3)
		{
			PrintHelp();
			goto done;
		}

		if (vgg16_net.loadnet(argv[2]) != 0)
		{
			_tprintf(_T("Failed to load the VGG network from %s.\n"), argv[2]);
			goto done;
		}

		vgg16_net.classify(argv[3]);
	}
	else
	{
		PrintHelp();
		goto done;
	}

done:
	CoUninitialize();

	return 0;
}

