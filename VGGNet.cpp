#include "VGGNet.h"
#include <torch/nn/module.h>
#include <iostream>
#include <tuple>
#include <chrono>
#include <io.h>
#include <tchar.h>
#include <random>
#include <algorithm>

extern void FreeBlob(void* p);

#define VGG_INPUT_IMG_WIDTH						224
#define VGG_INPUT_IMG_HEIGHT					224
#define VGG_TRAIN_BATCH_SIZE					64

VGGNet::VGGNet(int num_classes)
	: C1  (register_module("C1",  Conv2d(Conv2dOptions(  3,  64, 3).padding(1))))
	, C1B (register_module("C1B", BatchNorm2d(BatchNorm2dOptions(64))))
	, C3  (register_module("C3",  Conv2d(Conv2dOptions( 64,  64, 3).padding(1))))
	, C3B (register_module("C3B", BatchNorm2d(BatchNorm2dOptions(64))))
	, C6  (register_module("C6",  Conv2d(Conv2dOptions( 64, 128, 3).padding(1))))
	, C6B (register_module("C6B", BatchNorm2d(BatchNorm2dOptions(128))))
	, C8  (register_module("C8",  Conv2d(Conv2dOptions(128, 128, 3).padding(1))))
	, C8B (register_module("C8B", BatchNorm2d(BatchNorm2dOptions(128))))
	, C11 (register_module("C11", Conv2d(Conv2dOptions(128, 256, 3).padding(1))))
	, C11B(register_module("C11B",BatchNorm2d(BatchNorm2dOptions(256))))
	, C13 (register_module("C13", Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
	, C13B(register_module("C13B",BatchNorm2d(BatchNorm2dOptions(256))))
	, C15 (register_module("C15", Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
	, C15B(register_module("C15B",BatchNorm2d(BatchNorm2dOptions(256))))
	, C18 (register_module("C18", Conv2d(Conv2dOptions(256, 512, 3).padding(1))))
	, C18B(register_module("C18B",BatchNorm2d(BatchNorm2dOptions(512))))
	, C20 (register_module("C20", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C20B(register_module("C20B",BatchNorm2d(BatchNorm2dOptions(512))))
	, C22 (register_module("C22", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C22B(register_module("C22B",BatchNorm2d(BatchNorm2dOptions(512))))
	, C25 (register_module("C25", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C25B(register_module("C25B",BatchNorm2d(BatchNorm2dOptions(512))))
	, C27 (register_module("C27", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C27B(register_module("C27B",BatchNorm2d(BatchNorm2dOptions(512))))
	, C29 (register_module("C29", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C29B(register_module("C29B",BatchNorm2d(BatchNorm2dOptions(512))))
	, FC32(register_module("FC32",Linear(512 * 7 * 7, 4096)))
	, FC35(register_module("FC35",Linear(4096, 4096)))
	, FC38(register_module("FC38",Linear(4096, num_classes)))
{
	m_imageprocessor.Init(VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT);
}

VGGNet::~VGGNet()
{
	m_imageprocessor.Uninit();
}

int64_t VGGNet::num_flat_features(torch::Tensor input)
{
	int64_t num_features = 1;
	auto sizes = input.sizes();
	for (auto s : sizes) {
		num_features *= s;
	}
	return num_features;
}

torch::Tensor VGGNet::forward(torch::Tensor& x)
{
	namespace F = torch::nn::functional;

	if (m_bEnableBatchNorm)
	{
		// block#1
		x = F::max_pool2d(F::relu(C3B(C3(F::relu(C1B(C1(x)), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)), F::MaxPool2dFuncOptions(2));

		// block#2
		x = F::max_pool2d(F::relu(C8B(C8(F::relu(C6B(C6(x)), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)), F::MaxPool2dFuncOptions(2));

		// block#3
		x = F::max_pool2d(F::relu(C15B(C15(F::relu(C13B(C13(F::relu(C11B(C11(x)), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)), F::MaxPool2dFuncOptions(2));

		// block#4
		x = F::max_pool2d(F::relu(C22B(C22(F::relu(C20B(C20(F::relu(C18B(C18(x)), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)), F::MaxPool2dFuncOptions(2));

		// block#5
		x = F::max_pool2d(F::relu(C29B(C29(F::relu(C27B(C27(F::relu(C25B(C25(x)), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)))), F::ReLUFuncOptions(true)), F::MaxPool2dFuncOptions(2));
	}
	else
	{
		// block#1
		x = F::max_pool2d(F::relu(C3(F::relu(C1(x)))), F::MaxPool2dFuncOptions(2));

		// block#2
		x = F::max_pool2d(F::relu(C8(F::relu(C6(x)))), F::MaxPool2dFuncOptions(2));

		// block#3
		x = F::max_pool2d(F::relu(C15(F::relu(C13(F::relu(C11(x)))))), F::MaxPool2dFuncOptions(2));

		// block#4
		x = F::max_pool2d(F::relu(C22(F::relu(C20(F::relu(C18(x)))))), F::MaxPool2dFuncOptions(2));

		// block#5
		x = F::max_pool2d(F::relu(C29(F::relu(C27(F::relu(C25(x)))))), F::MaxPool2dFuncOptions(2));
	}

	x = x.view({ x.size(0), -1 });

	// classifier
	x = F::dropout(F::relu(FC32(x), F::ReLUFuncOptions(true)), F::DropoutFuncOptions().p(0.5).inplace(true));
	x = F::dropout(F::relu(FC35(x), F::ReLUFuncOptions(true)), F::DropoutFuncOptions().p(0.5).inplace(true));
	x = FC38(x);

	return x;
}

int VGGNet::train(const TCHAR* szImageSetRootPath, const TCHAR* szTrainSetStateFilePath)
{
	TCHAR szImageFile[MAX_PATH] = {0};
	// store the file name classname/picture_file_name
	std::vector<tstring> train_image_files;
	std::vector<tstring> train_image_labels;
	std::vector<size_t> train_image_shuffle_set;
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	TCHAR szDirPath[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (FAILED(loadImageSet(szImageSetRootPath, 
		train_image_files, train_image_labels, true)))
	{
		printf("Failed to load the train image/label set.\n");
		return -1;
	}

	double lr = 0.01;
	auto criterion = torch::nn::CrossEntropyLoss();
	//auto optimizer = torch::optim::SGD(parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));
	auto optimizer = torch::optim::SGD(parameters(), torch::optim::SGDOptions(lr).momentum(0.9));
	tm_end = std::chrono::system_clock::now();
	printf("It takes %lld msec to prepare training classifying cats and dogs.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	tm_start = std::chrono::system_clock::now();
	
	int64_t kNumberOfEpochs = 3;

	torch::Tensor tensor_input;
	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
	{
		auto running_loss = 0.;
		size_t totals = 0;

		// Shuffle the list
		if (train_image_files.size() > 0)
		{
			// generate the shuffle list to train
			train_image_shuffle_set.resize(train_image_files.size());
			for (size_t i = 0; i < train_image_files.size(); i++)
				train_image_shuffle_set[i] = i;
			std::random_device rd;
			std::mt19937_64 g(rd());
			std::shuffle(train_image_shuffle_set.begin(), train_image_shuffle_set.end(), g);
		}

		for (auto& pg : optimizer.param_groups())
		{
			if (pg.has_options())
			{
				auto& options = static_cast<torch::optim::SGDOptions&>(pg.options());
				options.lr() = lr;
			}
		}

		// Take the image shuffle
		for(size_t i = 0;i<(train_image_shuffle_set.size() + VGG_TRAIN_BATCH_SIZE -1)/ VGG_TRAIN_BATCH_SIZE;i++)
		{
			std::vector<VGGNet::tstring> image_batches;
			std::vector<long long> label_batches;

			for (int b = 0; b < VGG_TRAIN_BATCH_SIZE; b++)
			{
				size_t idx = i * VGG_TRAIN_BATCH_SIZE + b;
				if (idx >= train_image_shuffle_set.size())
					break;

				tstring& strImgFilePath = train_image_files[train_image_shuffle_set[idx]];
				const TCHAR* cszImgFilePath = strImgFilePath.c_str();
				const TCHAR* pszTmp = _tcschr(cszImgFilePath, _T('\\'));

				if (pszTmp == NULL)
					continue;

				size_t label = 0;
				for (label = 0; label < train_image_labels.size(); label++)
					if (_tcsnicmp(train_image_labels[label].c_str(), cszImgFilePath,
						(pszTmp - cszImgFilePath) / sizeof(TCHAR)) == 0)
						break;

				if (label >= train_image_labels.size())
					continue;

				_stprintf_s(szImageFile, _T("%s\\training_set\\%s"), szDirPath, cszImgFilePath);

				image_batches.push_back(szImageFile);
				label_batches.push_back((long long)label);
			}

			if (image_batches.size() == 0)
				continue;
			
			if (m_imageprocessor.ToTensor(image_batches, tensor_input) != 0)
				continue;

			//_tprintf(_T("now training %s for the file: %s.\n"), 
			//	train_image_labels[label].c_str(), cszImgFilePath);
			// Label在这里必须是一阶向量，里面元素必须是整数类型
			torch::Tensor tensor_label = torch::tensor(label_batches);
			//tensor_label = tensor_label.view({ 1, -1 });

			totals += label_batches.size();

			optimizer.zero_grad();
			// 喂数据给网络
			auto outputs = forward(tensor_input);

			//std::cout << "tensor_label:" << tensor_label << "\noutputs.sizes(): " << outputs << '\n';

			//std::cout << outputs << '\n';
			//std::cout << tensor_label << '\n';

			// 通过交叉熵计算损失
			auto loss = criterion(outputs, tensor_label);
			// 反馈给网络，调整权重参数进一步优化
			loss.backward();
			optimizer.step();

			running_loss += loss.item().toFloat();
			if ((i + 1) % 10 == 0)
			{
				printf("[%lld, %5zu] loss: %.3f\n", epoch, i + 1, running_loss / 10);
				running_loss = 0.;
			}
		}

		lr = lr * 0.1;
		if (lr < 0.00001)
			lr = 0.00001;
	}

	printf("Finish training!\n");

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to finish training VGG network!\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	m_image_labels = train_image_labels;
	savenet(szTrainSetStateFilePath);

	return 0;
}

void VGGNet::verify(const TCHAR* szImageSetRootPath, const TCHAR* szPreTrainSetStateFilePath)
{
	TCHAR szImageFile[MAX_PATH] = { 0 };
	// store the file name with the format 'classname/picture_file_name'
	std::vector<tstring> test_image_files;
	std::vector<tstring> test_image_labels;
	std::vector<size_t> test_image_shuffle_set;
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	TCHAR szDirPath[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (FAILED(loadImageSet(szImageSetRootPath, 
		test_image_files, test_image_labels, false)))
	{
		printf("Failed to load the test image/label sets.\n");
		return;
	}

	// Shuffle the list
	if (test_image_files.size() > 0)
	{
		// generate the shuffle list to train
		test_image_shuffle_set.resize(test_image_files.size());
		for (size_t i = 0; i < test_image_files.size(); i++)
			test_image_shuffle_set[i] = i;
		std::random_device rd;
		std::mt19937_64 g(rd());
		std::shuffle(test_image_shuffle_set.begin(), test_image_shuffle_set.end(), g);
	}

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to load the test images/labels set.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
	tm_start = std::chrono::system_clock::now();

	if (loadnet(szPreTrainSetStateFilePath) != 0)
	{
		printf("Failed to load the pre-trained VGG network.\n");
		return;
	}

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to load the pre-trained network state.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
	tm_start = std::chrono::system_clock::now();

	torch::Tensor tensor_input;
	int total_test_items = 0, passed_test_items = 0;
	for (size_t i = 0; i < test_image_shuffle_set.size(); i++)
	{
		tstring& strImgFilePath = test_image_files[test_image_shuffle_set[i]];
		const TCHAR* cszImgFilePath = strImgFilePath.c_str();
		const TCHAR* pszTmp = _tcschr(cszImgFilePath, _T('\\'));

		if (pszTmp == NULL)
			continue;

		size_t label = 0;
		for (label = 0; label < m_image_labels.size(); label++)
			if (_tcsnicmp(m_image_labels[label].c_str(), 
				cszImgFilePath, (pszTmp - cszImgFilePath) / sizeof(TCHAR)) == 0)
				break;

		if (label >= m_image_labels.size())
		{
			tstring strUnmatchedLabel(cszImgFilePath, (pszTmp - cszImgFilePath) / sizeof(TCHAR));
			_tprintf(_T("Can't find the test label: %s\n"), strUnmatchedLabel.c_str());
			continue;
		}

		_stprintf_s(szImageFile, _T("%s\\test_set\\%s"), szDirPath, cszImgFilePath);
		if (m_imageprocessor.ToTensor(szImageFile, tensor_input) != 0)
			continue;

		// Label在这里必须是一阶向量，里面元素必须是整数类型
		torch::Tensor tensor_label = torch::tensor({ (int64_t)label });

		auto outputs = forward(tensor_input);
		auto predicted = torch::max(outputs, 1);

		//_tprintf(_T("predicted: %s, fact: %s --> file: %s.\n"), 
		//	m_image_labels[std::get<1>(predicted).item<int>()].c_str(), 
		//	m_image_labels[tensor_label[0].item<int>()].c_str(), szImageFile);

		if (tensor_label[0].item<int>() == std::get<1>(predicted).item<int>())
			passed_test_items++;

		total_test_items++;

		//printf("label: %d.\n", labels[0].item<int>());
		//printf("predicted label: %d.\n", std::get<1>(predicted).item<int>());
		//std::cout << std::get<1>(predicted) << '\n';

		//break;
	}
	tm_end = std::chrono::system_clock::now();

	printf("Total test items: %d, passed test items: %d, pass rate: %.3f%%, cost %lld msec.\n",
		total_test_items, passed_test_items, passed_test_items*100.f / total_test_items,
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
}

void VGGNet::classify(const TCHAR* cszImageFile)
{
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;
	torch::Tensor tensor_input;

	if (m_imageprocessor.ToTensor(cszImageFile, tensor_input) != 0)
	{
		printf("Failed to convert the image to tensor.\n");
		return;
	}

	auto outputs = forward(tensor_input);
	auto predicted = torch::max(outputs, 1);

	tm_end = std::chrono::system_clock::now();

	_tprintf(_T("This image seems to %s, cost %lld msec.\n"),
		m_image_labels.size() > std::get<1>(predicted).item<int>()?m_image_labels[std::get<1>(predicted).item<int>()].c_str():_T("Unknown"),
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
}

HRESULT VGGNet::loadImageSet(
	const TCHAR* szRootPath,				// the root path to place training_set or test_set folder
	std::vector<tstring>& image_files,		// the image files to be trained or tested
	std::vector<tstring>& image_labels,		// the image label
	bool bTrainSet, bool bShuffle)
{
	HRESULT hr = S_OK;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szImageFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szRootPath);
	size_t ccDirPath = _tcslen(szRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szImageFile, MAX_PATH, _T("%s\\%s\\*.*"), 
		szDirPath, bTrainSet ? _T("training_set") : _T("test_set"));

	// Find all image file names under the train set, 2 level
	WIN32_FIND_DATA find_data;
	HANDLE hFind = FindFirstFile(szImageFile, &find_data);
	if (hFind == INVALID_HANDLE_VALUE)
		return E_FAIL;

	do {
		if (!(find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) ||
			_tcsicmp(find_data.cFileName, _T(".")) == 0 ||
			_tcsicmp(find_data.cFileName, _T("..")) == 0)
			continue;

		WIN32_FIND_DATA image_find_data;
		_stprintf_s(szImageFile, MAX_PATH, _T("%s\\%s\\%s\\*.*"), 
			szDirPath, bTrainSet?_T("training_set"):_T("test_set"), find_data.cFileName);

		BOOL bHaveTrainImages = FALSE;
		HANDLE hImgFind = FindFirstFile(szImageFile, &image_find_data);
		if (hImgFind == INVALID_HANDLE_VALUE)
			continue;

		do
		{
			if (image_find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)
				continue;

			// check whether it is a supported image file
			const TCHAR* szTmp = _tcsrchr(image_find_data.cFileName, _T('.'));
			if (szTmp && (_tcsicmp(szTmp, _T(".jpg")) == 0 ||
				_tcsicmp(szTmp, _T(".png")) == 0 ||
				_tcsicmp(szTmp, _T(".jpeg")) == 0))
			{
				// reuse szImageFile
				_stprintf_s(szImageFile, _T("%s\\%s"), find_data.cFileName, image_find_data.cFileName);
				image_files.emplace_back(szImageFile);
				if (bHaveTrainImages == FALSE)
				{
					bHaveTrainImages = TRUE;
					image_labels.emplace_back(find_data.cFileName);
				}
			}

		} while (FindNextFile(hImgFind, &image_find_data));

		FindClose(hImgFind);

	} while (FindNextFile(hFind, &find_data));

	FindClose(hFind);

	return hr;
}

HRESULT VGGNet::loadLabels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels)
{
	HRESULT hr = S_OK;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szImageFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szImageFile, MAX_PATH, _T("%s\\training_set\\*.*"), szDirPath);

	// Find all image file names under the train set, 2 level
	WIN32_FIND_DATA find_data;
	HANDLE hFind = FindFirstFile(szImageFile, &find_data);
	if (hFind == INVALID_HANDLE_VALUE)
		return E_FAIL;

	do {
		if (!(find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY))
			continue;

		WIN32_FIND_DATA image_find_data;
		_stprintf_s(szImageFile, MAX_PATH, _T("%s\\training_set\\%s\\*.*"), szDirPath, find_data.cFileName);

		BOOL bHaveTrainImages = FALSE;
		HANDLE hImgFind = FindFirstFile(szImageFile, &image_find_data);
		if (hImgFind == INVALID_HANDLE_VALUE)
			continue;

		do
		{
			if (image_find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)
				continue;

			// check whether it is a supported image file
			const TCHAR* szTmp = _tcsrchr(image_find_data.cFileName, _T('.'));
			if (szTmp && (_tcsicmp(szTmp, _T(".jpg")) == 0 ||
				_tcsicmp(szTmp, _T(".png")) == 0 ||
				_tcsicmp(szTmp, _T(".jpeg")) == 0))
			{
				bHaveTrainImages = TRUE;
				break;
			}

		} while (FindNextFile(hImgFind, &image_find_data));

		if (bHaveTrainImages)
			image_labels.emplace_back(find_data.cFileName);

		FindClose(hImgFind);

	} while (FindNextFile(hFind, &find_data));

	FindClose(hFind);

	return S_OK;
}

int VGGNet::savenet(const TCHAR* szTrainSetStateFilePath)
{
	// Save the net state to xxxx.pt and save the labels to xxxx.pt.label
	std::string szLabelFilePath;
	try
	{
		torch::serialize::OutputArchive archive;
		save(archive);

#ifdef _UNICODE
		char szInputFile[MAX_PATH + 1] = { 0 };
		if (WideCharToMultiByte(CP_UTF8, 0, szTrainSetStateFilePath, -1, szInputFile, MAX_PATH + 1, NULL, NULL) <= 0)
		{
			printf("Failed to convert the file name to UTF-8.\n");
			return -1;
		}
		else
			archive.save_to(szInputFile);
		szLabelFilePath = szInputFile;
#else
		archive.save_to(szTrainSetStateFilePath);
		szLabelFilePath = szInputFile;
#endif
	}
	catch (...)
	{
		printf("Failed to save the trained VGG net state.\n");
		return -1;
	}
	_tprintf(_T("Save the training result to %s.\n"), szTrainSetStateFilePath);

	// write the labels
	szLabelFilePath.append(".label");
	char szLabel[MAX_LABEL_NAME] = { 0 };
	try
	{
		torch::serialize::OutputArchive label_archive;
		c10::List<std::string> label_list;
		for (size_t i = 0; i < m_image_labels.size(); i++)
		{
			memset(szLabel, 0, sizeof(szLabel));
			WideCharToMultiByte(CP_UTF8, 0, m_image_labels[i].c_str(), -1, szLabel, MAX_LABEL_NAME, NULL, NULL);
			label_list.emplace_back((const char*)szLabel);
		}
		torch::IValue value(label_list);
		label_archive.write("labels", label_list);
		label_archive.save_to(szLabelFilePath);
	}
	catch (...)
	{
		printf("Failed to save the label names.\n");
		return -1;
	}

	printf("Save the supported labels to be classified to %s.\n", szLabelFilePath.c_str());
	return 0;
}

int VGGNet::loadnet(const TCHAR* szTrainSetStateFilePath)
{
	std::string szLabelFilePath;
	try
	{
		torch::serialize::InputArchive archive;

#ifdef _UNICODE
		char szInputFile[MAX_PATH + 1] = { 0 };
		if (WideCharToMultiByte(CP_UTF8, 0, szTrainSetStateFilePath, -1, szInputFile, MAX_PATH + 1, NULL, NULL) <= 0)
		{
			printf("Failed to convert the file name.\n");
			return -1;
		}
		else
			archive.load_from(szInputFile);
		szLabelFilePath = szInputFile;
#else
		archive.load_from(szPreTrainSetStateFilePath);
		szLabelFilePath = szPreTrainSetStateFilePath;
#endif
		load(archive);
	}
	catch (...)
	{
		printf("Failed to load the pre-trained VGG net state.\n");
		return -1;
	}

	// load the label files for this pre-trained network;
	wchar_t szLabel[MAX_LABEL_NAME] = { 0 };
	szLabelFilePath.append(".label");
	m_image_labels.clear();
	try
	{
		torch::IValue value;
		torch::serialize::InputArchive label_archive;
		label_archive.load_from(szLabelFilePath.c_str());

		if (label_archive.try_read("labels", value) && value.isList())
		{
			auto& label_list = value.toListRef();
			for (size_t i = 0; i < label_list.size(); i++)
			{
#ifdef _UNICODE
				if (MultiByteToWideChar(CP_UTF8, 0, label_list[i].toStringRef().c_str(), -1, szLabel, MAX_LABEL_NAME) <= 0)
					m_image_labels.push_back(_T("Unknown"));
				else
					m_image_labels.push_back(szLabel);
#else
				m_image_labels.push_back(label_list.get(i).toStringRef());
#endif
			}
		}
	}
	catch (...)
	{
		printf("Failed to save the label names.\n");
		return -1;
	}

	printf("Load the supported labels to be classified to %s.\n", szLabelFilePath.c_str());
	return 0;
}

