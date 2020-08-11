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

VGGNet::VGGNet(int num_classes)
	: C1  (register_module("C1",  Conv2d(Conv2dOptions(  3,  64, 3).padding(1))))
	, C3  (register_module("C3",  Conv2d(Conv2dOptions( 64,  64, 3).padding(1))))
	, C6  (register_module("C6",  Conv2d(Conv2dOptions( 64, 128, 3).padding(1))))
	, C8  (register_module("C8",  Conv2d(Conv2dOptions(128, 128, 3).padding(1))))
	, C11 (register_module("C11", Conv2d(Conv2dOptions(128, 256, 3).padding(1))))
	, C13 (register_module("C13", Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
	, C15 (register_module("C15", Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
	, C18 (register_module("C18", Conv2d(Conv2dOptions(256, 512, 3).padding(1))))
	, C20 (register_module("C20", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C22 (register_module("C22", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C25 (register_module("C25", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C27 (register_module("C27", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, C29 (register_module("C29", Conv2d(Conv2dOptions(512, 512, 3).padding(1))))
	, FC32(register_module("FC32",Linear(512 * 7 * 7, 4096)))
	, FC35(register_module("FC35",Linear(4096, 4096)))
	, FC38(register_module("FC38",Linear(4096, num_classes)))
{
	HRESULT hr = S_OK;

	// Create D2D1 factory to create the related render target and D2D1 objects
	D2D1_FACTORY_OPTIONS options;
	ZeroMemory(&options, sizeof(D2D1_FACTORY_OPTIONS));
#if defined(_DEBUG)
	// If the project is in a debug build, enable Direct2D debugging via SDK Layers.
	options.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif
	if (SUCCEEDED(hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED,
		__uuidof(ID2D1Factory2), &options, &m_spD2D1Factory)))
	{
		// Create the image factory
		if (SUCCEEDED(hr = CoCreateInstance(CLSID_WICImagingFactory,
			nullptr, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&m_spWICImageFactory)))
		{
			// 创建一个Pre-multiplexed BGRA的224x224的WICBitmap
			if (SUCCEEDED(hr = m_spWICImageFactory->CreateBitmap(VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT, GUID_WICPixelFormat32bppPBGRA,
				WICBitmapCacheOnDemand, &m_spNetInputBitmap)))
			{
				// 在此WICBitmap上创建D2D1 Render Target
				D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT,
					D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), 96, 96);
				if (SUCCEEDED(hr = m_spD2D1Factory->CreateWicBitmapRenderTarget(m_spNetInputBitmap.Get(), props, &m_spRenderTarget)))
				{
					hr = m_spRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Black, 1.0f), &m_spBGBrush);
				}
			}
		}
	}

	m_pBGRABuf = new unsigned char[VGG_INPUT_IMG_WIDTH*VGG_INPUT_IMG_HEIGHT * 4];
}

VGGNet::~VGGNet()
{
	delete[] m_pBGRABuf;
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

bool VGGNet::GetImageDrawRect(UINT target_width, UINT target_height, UINT image_width, UINT image_height, D2D1_RECT_F& dst_rect)
{
	if (target_width == 0 || target_height == 0 || image_width == 0 || image_height == 0)
		return false;

	if (target_width*image_height >= image_width*target_height)
	{
		// align with height
		FLOAT ratio = (FLOAT)target_height / image_height;
		dst_rect.top = 0;
		dst_rect.bottom = (FLOAT)target_height;
		dst_rect.left = (target_width - image_width * ratio) / 2.0f;
		dst_rect.right = (target_width + image_width * ratio) / 2.0f;
	}
	else
	{
		// align with width
		FLOAT ratio = (FLOAT)target_width / image_width;
		dst_rect.left = 0;
		dst_rect.right = (FLOAT)target_width;
		dst_rect.top = (target_height - image_height * ratio) / 2.0f;
		dst_rect.bottom = (target_height + image_height * ratio) / 2.0f;
	}

	return true;
}

torch::Tensor VGGNet::forward(torch::Tensor input)
{
	namespace F = torch::nn::functional;

	// block#1
	auto x = F::max_pool2d(F::relu(C3(F::relu(C1(input)))), F::MaxPool2dFuncOptions(2));

	// block#2
	x = F::max_pool2d(F::relu(C8(F::relu(C6(x)))), F::MaxPool2dFuncOptions(2));

	// block#3
	x = F::max_pool2d(F::relu(C15(F::relu(C13(F::relu(C11(x)))))), F::MaxPool2dFuncOptions(2));

	// block#4
	x = F::max_pool2d(F::relu(C22(F::relu(C20(F::relu(C18(x)))))), F::MaxPool2dFuncOptions(2));

	// block#5
	x = F::max_pool2d(F::relu(C29(F::relu(C27(F::relu(C25(x)))))), F::MaxPool2dFuncOptions(2));

	x = x.view({ -1, num_flat_features(x) });

	// classifier
	x = F::dropout(F::relu(FC32(x)), F::DropoutFuncOptions().p(0.5));
	x = F::dropout(F::relu(FC35(x)), F::DropoutFuncOptions().p(0.5));
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

	if (FAILED(loadImageSet(szImageSetRootPath, train_image_files, train_image_labels, train_image_shuffle_set, true)))
	{
		printf("Failed to load the train image/label set.\n");
		return -1;
	}

	auto criterion = torch::nn::CrossEntropyLoss();
	auto optimizer = torch::optim::SGD(parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));
	tm_end = std::chrono::system_clock::now();
	printf("It takes %lld msec to prepare training classifying cats and dogs.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	tm_start = std::chrono::system_clock::now();
	
	int64_t kNumberOfEpochs = 2;

	torch::Tensor tensor_input;
	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
	{
		auto running_loss = 0.;
		// Take the image shuffle
		for(size_t i = 0;i<train_image_shuffle_set.size();i++)
		{
			tstring& strImgFilePath = train_image_files[train_image_shuffle_set[i]];
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
			if (toTensor(szImageFile, tensor_input) != 0)
				continue;

			//_tprintf(_T("now training %s for the file: %s.\n"), 
			//	train_image_labels[label].c_str(), cszImgFilePath);
			// Label在这里必须是一阶向量，里面元素必须是整数类型
			torch::Tensor tensor_label = torch::tensor({ (int64_t)label });

			optimizer.zero_grad();
			// 喂数据给网络
			auto outputs = forward(tensor_input);

			//std::cout << outputs << '\n';
			//std::cout << tensor_label << '\n';

			// 通过交叉熵计算损失
			auto loss = criterion(outputs, tensor_label);
			// 反馈给网络，调整权重参数进一步优化
			loss.backward();
			optimizer.step();

			running_loss += loss.item().toFloat();
			if ((i + 1) % 100 == 0)
			{
				printf("[%lld, %5zu] loss: %.3f\n", epoch, i + 1, running_loss / 100);
				running_loss = 0.;
			}
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
		test_image_files, test_image_labels, test_image_shuffle_set, false)))
	{
		printf("Failed to load the test image/label sets.\n");
		return;
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
		if (toTensor(szImageFile, tensor_input) != 0)
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

	if (toTensor(cszImageFile, tensor_input) != 0)
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

HRESULT VGGNet::toTensor(const TCHAR* cszImageFile, torch::Tensor& tensor)
{
	HRESULT hr = S_OK;
	ComPtr<IWICBitmapDecoder> spDecoder;				// Image decoder
	ComPtr<IWICBitmapFrameDecode> spBitmapFrameDecode;	// Decoded image
	ComPtr<IWICBitmapSource> spConverter;				// Converted image
	ComPtr<IWICBitmap> spHandWrittenBitmap;				// The original bitmap
	ComPtr<ID2D1Bitmap> spD2D1Bitmap;					// D2D1 bitmap

	UINT uiFrameCount = 0;
	UINT uiWidth = 0, uiHeight = 0;
	WICPixelFormatGUID pixelFormat;
	D2D1_RECT_F dst_rect = { 0, 0, VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT };
	WICRect rect = { 0, 0, VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT };

	if (cszImageFile == NULL || _taccess(cszImageFile, 0) != 0)
		return E_INVALIDARG;

	wchar_t* wszInputFile = NULL;
	size_t cbFileName = _tcslen(cszImageFile);
#ifndef _UNICODE
	wszInputFile = new wchar_t[cbFileName + 1];
	if (MultiByteToWideChar(CP_UTF8, 0, cszCatImageFile, -1, wszInputFile, cbFileName + 1) == 0)
	{
		delete[] wszInputFile;
		return -1;
	}
#else
	wszInputFile = (wchar_t*)cszImageFile;
#endif

	// 加载图片, 并为其创建图像解码器
	if (FAILED(m_spWICImageFactory->CreateDecoderFromFilename(wszInputFile, NULL,
		GENERIC_READ, WICDecodeMetadataCacheOnDemand, &spDecoder)))
		goto done;

	// 得到多少帧图像在图片文件中，如果无可解帧，结束程序
	if (FAILED(hr = spDecoder->GetFrameCount(&uiFrameCount)) || uiFrameCount == 0)
		goto done;

	// 得到第一帧图片
	if (FAILED(hr = hr = spDecoder->GetFrame(0, &spBitmapFrameDecode)))
		goto done;

	// 得到图片大小
	if (FAILED(hr = spBitmapFrameDecode->GetSize(&uiWidth, &uiHeight)))
		goto done;

	// 得到图片像素格式
	if (FAILED(hr = spBitmapFrameDecode->GetPixelFormat(&pixelFormat)))
		goto done;

	// 如果图片不是Pre-multiplexed BGRA格式，转化成这个格式，以便用D2D硬件处理图形转换
	if (!IsEqualGUID(pixelFormat, GUID_WICPixelFormat32bppPBGRA))
	{
		if (FAILED(hr = WICConvertBitmapSource(GUID_WICPixelFormat32bppPBGRA,
			spBitmapFrameDecode.Get(), &spConverter)))
			goto done;
	}
	else
		spConverter = spBitmapFrameDecode;

	// 转化为Pre-multiplexed BGRA格式的WICBitmap
	if (FAILED(hr = m_spWICImageFactory->CreateBitmapFromSource(
		spConverter.Get(), WICBitmapCacheOnDemand, &spHandWrittenBitmap)))
		goto done;

	// 将转化为Pre-multiplexed BGRA格式的WICBitmap的原始图片转换到D2D1Bitmap对象中来，以便后面的缩放处理
	if (FAILED(hr = m_spRenderTarget->CreateBitmapFromWicBitmap(spHandWrittenBitmap.Get(), &spD2D1Bitmap)))
		goto done;

	// 将图片进行缩放处理，转化为224x224的图片
	m_spRenderTarget->BeginDraw();
		
	m_spRenderTarget->FillRectangle(dst_rect, m_spBGBrush.Get());

	if (GetImageDrawRect(VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT, uiWidth, uiHeight, dst_rect))
		m_spRenderTarget->DrawBitmap(spD2D1Bitmap.Get(), &dst_rect);
	
	m_spRenderTarget->EndDraw();

	//ImageProcess::SaveAs(m_spNetInputBitmap, L"I:\\test.png");

	// 并将图像每个channel中数据转化为[-1.0, 1.0]的raw data
	hr = m_spNetInputBitmap->CopyPixels(&rect, VGG_INPUT_IMG_WIDTH * 4, 4 * VGG_INPUT_IMG_WIDTH * VGG_INPUT_IMG_HEIGHT, m_pBGRABuf);

	float* res_data = (float*)malloc(3 * VGG_INPUT_IMG_WIDTH * VGG_INPUT_IMG_HEIGHT * sizeof(float));
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < VGG_INPUT_IMG_HEIGHT; i++)
		{
			for (int j = 0; j < VGG_INPUT_IMG_WIDTH; j++)
			{
				int pos = c * VGG_INPUT_IMG_WIDTH*VGG_INPUT_IMG_HEIGHT + i * VGG_INPUT_IMG_WIDTH + j;
				res_data[pos] = ((255 - m_pBGRABuf[i * VGG_INPUT_IMG_WIDTH * 4 + j * 4 + 2 - c]) / 255.0f - 0.5f) / 0.5f;
			}
		}
	}

	tensor = torch::from_blob(res_data, { 1, 3, VGG_INPUT_IMG_WIDTH, VGG_INPUT_IMG_HEIGHT }, FreeBlob);

	hr = S_OK;

done:
	if (wszInputFile != NULL && wszInputFile != cszImageFile)
		delete[] wszInputFile;
	return hr;
}

HRESULT VGGNet::loadImageSet(
	const TCHAR* szRootPath,				// the root path to place training_set or test_set folder
	std::vector<tstring>& image_files,		// the image files to be trained or tested
	std::vector<tstring>& image_labels,		// the image label
	std::vector<size_t>& image_shuffle_set,	// the shuffle image set, ex, [1, 0, 3, 4, 2]
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

	if (image_files.size() > 0)
	{
		// generate the shuffle list to train
		image_shuffle_set.resize(image_files.size());
		for (size_t i = 0; i < image_files.size(); i++)
			image_shuffle_set[i] = i;
		std::random_device rd;
		std::mt19937_64 g(rd());
		std::shuffle(image_shuffle_set.begin(), image_shuffle_set.end(), g);
	}

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

	printf("Save the supported labels to be classified to %s.\n", szLabelFilePath.c_str());
	return 0;
}

