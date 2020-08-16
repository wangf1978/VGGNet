#include "VGGNet.h"
#include <torch/nn/module.h>
#include <iostream>
#include <tuple>
#include <chrono>
#include <io.h>
#include <tchar.h>
#include <random>
#include <algorithm>
#include "util.h"

extern void FreeBlob(void* p);

#define VGG_INPUT_IMG_WIDTH						224
#define VGG_INPUT_IMG_HEIGHT					224
#define VGG_TRAIN_BATCH_SIZE					64

std::map<VGG_CONFIG, std::string> _VGG_CONFIG_NAMES =
{
	{VGG_A,					"VGGA_NoBatchNorm"},
	{VGG_A_BATCHNORM,		"VGGA_BatchNorm"},
	{VGG_A_LRN,				"VGGA_LRN_NoBatchNorm"},
	{VGG_A_LRN_BATCHNORM,	"VGGA_LRN_BatchNorm"},
	{VGG_B,					"VGGB_NoBatchNorm"},
	{VGG_B_BATCHNORM,		"VGGB_BatchNorm"},
	{VGG_C,					"VGGC_NoBatchNorm"},
	{VGG_C_BATCHNORM,		"VGGC_BatchNorm"},
	{VGG_D,					"VGGD_NoBatchNorm"},
	{VGG_D_BATCHNORM,		"VGGD_BatchNorm"},
	{VGG_E,					"VGGD_NoBatchNorm"},
	{VGG_E_BATCHNORM,		"VGGD_BatchNorm"},
};

VGGNet::VGGNet(VGG_CONFIG config, int num_classes, bool use_32x32_input, int * ret)
	: m_VGG_config(config)
	, m_num_classes(num_classes)
	, m_use_32x32_input(use_32x32_input)
{
	m_imageprocessor.Init(use_32x32_input ? 32 : VGG_INPUT_IMG_WIDTH, use_32x32_input ? 32 : VGG_INPUT_IMG_HEIGHT);

	int iRet = -1;
	auto iter = _VGG_CONFIG_NAMES.find(config);
	if (iter != _VGG_CONFIG_NAMES.end())
	{
		SetOptions({ 
			{"NN::final_out_classes", std::to_string(m_num_classes) },
			{"NN::use_32x32_input", std::to_string(m_use_32x32_input?1:0) },
			});
		iRet = Init(iter->second.c_str());

		if (iRet == 0 && use_32x32_input)
		{
			for (auto& m : modules(false))
			{
				if (m->name() == "module: torch::nn::LinearImpl")
				{
					std::shared_ptr<torch::nn::LinearImpl> spLinearImpl =
						std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
					spLinearImpl->options.in_features(512);
					break;
				}
			}
		}
	}

	if (ret != NULL)
		*ret = iRet;
}

VGGNet::~VGGNet()
{
	m_imageprocessor.Uninit();
	Uninit();
}

int VGGNet::train(const char* szImageSetRootPath, 
	const char* szTrainSetStateFilePath,
	int batch_size,
	int num_epoch,
	float learning_rate,
	unsigned int showloss_per_num_of_batches,
	bool clean_pretrain_net)
{
	TCHAR szImageFile[MAX_PATH] = {0};
	// store the file name classname/picture_file_name
	std::vector<tstring> train_image_files;
	std::vector<tstring> train_image_labels;
	std::vector<size_t> train_image_shuffle_set;
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR* tszImageSetRootPath = NULL;

	// Convert UTF-8 to Unicode
#ifdef _UNICODE
	wchar_t wszImageSetRootPath[MAX_PATH + 1] = { 0 };
	MultiByteToWideChar(CP_UTF8, 0, szImageSetRootPath, -1, wszImageSetRootPath, MAX_PATH + 1);
	tszImageSetRootPath = wszImageSetRootPath;
#else
	tszImageSetRootPath = szImageSetRootPath;
#endif

	_tcscpy_s(szDirPath, MAX_PATH, tszImageSetRootPath);
	size_t ccDirPath = _tcslen(tszImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (FAILED(loadImageSet(tszImageSetRootPath,
		train_image_files, train_image_labels, true)))
	{
		printf("Failed to load the train image/label set.\n");
		return -1;
	}

	// delete the previous pre-train net state
	if (clean_pretrain_net)
	{
		if (DeleteFileA(szTrainSetStateFilePath) == FALSE)
		{
			printf("Failed to delete the file '%s' {err: %lu}.\n", szTrainSetStateFilePath, GetLastError());
		}
	}

	batch_size = batch_size < 0 ? 1 : batch_size;

	double lr = learning_rate > 0.f ? (double)learning_rate : 0.01;
	auto criterion = torch::nn::CrossEntropyLoss();
	//auto optimizer = torch::optim::SGD(parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));
	auto optimizer = torch::optim::SGD(parameters(), torch::optim::SGDOptions(lr).momentum(0.9));
	tm_end = std::chrono::system_clock::now();
	printf("It takes %lld msec to prepare training classifying cats and dogs.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	tm_start = std::chrono::system_clock::now();
	
	torch::Tensor tensor_input;
	for (int64_t epoch = 1; epoch <= (int64_t)num_epoch; ++epoch)
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
		for(size_t i = 0;i<(train_image_shuffle_set.size() + batch_size -1)/ batch_size;i++)
		{
			std::vector<VGGNet::tstring> image_batches;
			std::vector<long long> label_batches;

			for (int b = 0; b < batch_size; b++)
			{
				size_t idx = i * batch_size + b;
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
			if (showloss_per_num_of_batches > 0 && ((i + 1) % showloss_per_num_of_batches == 0))
			{
				printf("[%lld, %5zu] loss: %.3f\n", epoch, i + 1, running_loss / showloss_per_num_of_batches);
				running_loss = 0.;
			}
		}

		if (epoch % 2 == 0)
		{
			lr = lr * 0.1;
			if (lr < 0.00001)
				lr = 0.00001;
		}
	}

	printf("Finish training!\n");

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to finish training VGG network!\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	m_image_labels = train_image_labels;
	savenet(szTrainSetStateFilePath);

	return 0;
}

void VGGNet::verify(const char* szImageSetRootPath, const char* szPreTrainSetStateFilePath)
{
	TCHAR szImageFile[MAX_PATH] = { 0 };
	// store the file name with the format 'classname/picture_file_name'
	std::vector<tstring> test_image_files;
	std::vector<tstring> test_image_labels;
	std::vector<size_t> test_image_shuffle_set;
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR* tszImageSetRootPath = NULL;

	// Convert UTF-8 to Unicode
#ifdef _UNICODE
	wchar_t wszImageSetRootPath[MAX_PATH + 1] = { 0 };
	MultiByteToWideChar(CP_UTF8, 0, szImageSetRootPath, -1, wszImageSetRootPath, MAX_PATH + 1);
	tszImageSetRootPath = wszImageSetRootPath;
#else
	tszImageSetRootPath = szImageSetRootPath;
#endif

	_tcscpy_s(szDirPath, MAX_PATH, tszImageSetRootPath);
	size_t ccDirPath = _tcslen(tszImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (FAILED(loadImageSet(tszImageSetRootPath,
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

void VGGNet::classify(const char* cszImageFile)
{
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;
	torch::Tensor tensor_input;
	TCHAR *tcsImageFile;

#ifdef _UNICODE
	wchar_t wszImageFilePath[MAX_PATH + 1] = { 0 };
	MultiByteToWideChar(CP_UTF8, 0, cszImageFile, -1, wszImageFilePath, MAX_PATH + 1);
	tcsImageFile = wszImageFilePath;
#else
	tcsImageFile = cszImageFile;
#endif

	if (m_imageprocessor.ToTensor(tcsImageFile, tensor_input) != 0)
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

int VGGNet::savenet(const char* szTrainSetStateFilePath)
{
	// Save the net state to xxxx.pt and save the labels to xxxx.pt.label
	std::string szLabelFilePath;
	try
	{
		torch::serialize::OutputArchive archive;
		save(archive);

		archive.save_to(szTrainSetStateFilePath);
		szLabelFilePath = szTrainSetStateFilePath;
	}
	catch (...)
	{
		printf("Failed to save the trained VGG net state.\n");
		return -1;
	}
	printf("Save the training result to %s.\n", szTrainSetStateFilePath);

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

int VGGNet::loadnet(const char* szPreTrainSetStateFilePath)
{
	std::string szLabelFilePath;
	try
	{
		torch::serialize::InputArchive archive;

		archive.load_from(szPreTrainSetStateFilePath);
		szLabelFilePath = szPreTrainSetStateFilePath;
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

