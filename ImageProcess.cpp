#include "ImageProcess.h"
#include <stdio.h>
#include <io.h>
#include <tchar.h>

extern void FreeBlob(void* p);

ImageProcess::ImageProcess()
{
	HRESULT hr = S_OK;

	// Create D2D1 factory to create the related render target and D2D1 objects
	D2D1_FACTORY_OPTIONS options;
	ZeroMemory(&options, sizeof(D2D1_FACTORY_OPTIONS));
#if defined(_DEBUG)
	// If the project is in a debug build, enable Direct2D debugging via SDK Layers.
	options.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif
	if (FAILED(hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED,
		__uuidof(ID2D1Factory2), &options, &m_spD2D1Factory)))
		printf("Failed to create D2D1 factory {hr: 0X%X}.\n", hr);

	// Create the image factory
	if (FAILED(hr = CoCreateInstance(CLSID_WICImagingFactory,
		nullptr, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&m_spWICImageFactory)))
		printf("Failed to create WICImaging Factory {hr: 0X%X}.\n", hr);
}

ImageProcess::~ImageProcess()
{

}

HRESULT ImageProcess::Init(UINT outWidth, UINT outHeight)
{
	HRESULT hr = S_OK;
	if (outWidth == 0 || outHeight == 0)
	{
		// Use the original image width and height as the output width and height
		m_outWidth = outWidth;
		m_outHeight = outHeight;
		return hr;
	}

	// 创建一个Pre-multiplexed BGRA的224x224的WICBitmap
	if (SUCCEEDED(hr = m_spWICImageFactory->CreateBitmap(outWidth, outHeight, GUID_WICPixelFormat32bppPBGRA,
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

	// Create a buffer to be used for converting ARGB to tensor
	if (SUCCEEDED(hr))
	{
		if (m_pBGRABuf != NULL)
			delete[] m_pBGRABuf;
		m_pBGRABuf = new unsigned char[outWidth*outHeight * 4];
		m_outWidth = outWidth;
		m_outHeight = outHeight;
	}

	return hr;
}

void ImageProcess::Uninit()
{
	if (m_pBGRABuf != NULL)
	{
		delete[] m_pBGRABuf;
		m_pBGRABuf = NULL;
	}
}

void ImageProcess::SetRGBMeansAndStds(float means[3], float stds[3])
{
	for (int i = 0; i < 3; i++)
	{
		m_RGB_means[i] = means[i];
		m_RGB_stds[i] = stds[i];
	}
}

void ImageProcess::SetGreyScaleMeanAndStd(float mean, float std)
{
	m_GreyScale_mean = mean;
	m_GreyScale_std = std;
}

bool ImageProcess::GetImageDrawRect(UINT target_width, UINT target_height, UINT image_width, UINT image_height, D2D1_RECT_F& dst_rect)
{
	if (target_width == 0 || target_height == 0 || image_width == 0 || image_height == 0)
		return false;

	if (target_width*image_height >= image_width * target_height)
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

HRESULT ImageProcess::ToTensor(const TCHAR* cszImageFile, torch::Tensor& tensor)
{
	HRESULT hr = S_OK;
	ComPtr<IWICBitmapDecoder> spDecoder;				// Image decoder
	ComPtr<IWICBitmapFrameDecode> spBitmapFrameDecode;	// Decoded image
	ComPtr<IWICBitmapSource> spConverter;				// Converted image
	ComPtr<IWICBitmap> spHandWrittenBitmap;				// The original bitmap
	ComPtr<ID2D1Bitmap> spD2D1Bitmap;					// D2D1 bitmap

	ComPtr<IWICBitmap> spNetInputBitmap = m_spNetInputBitmap;
	ComPtr<ID2D1RenderTarget> spRenderTarget = m_spRenderTarget;
	ComPtr<ID2D1SolidColorBrush> spBGBrush = m_spBGBrush;

	BOOL bDynamic = FALSE;
	UINT uiFrameCount = 0;
	UINT uiWidth = 0, uiHeight = 0;
	UINT outWidth = m_outWidth;
	UINT outHeight = m_outHeight;
	WICPixelFormatGUID pixelFormat;
	unsigned char* pBGRABuf = m_pBGRABuf;
	D2D1_RECT_F dst_rect = { 0, 0, outWidth, outHeight };
	WICRect rect = { 0, 0, outWidth, outHeight };

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

	// 调整转换和输出
	if (outWidth == 0)
	{
		outWidth = uiWidth;
		dst_rect.right = uiWidth;
		rect.Width = uiWidth;
		bDynamic = TRUE;
	}

	if (outHeight == 0)
	{
		outHeight = uiHeight;
		dst_rect.bottom = uiHeight;
		rect.Height = uiHeight;
		bDynamic = TRUE;
	}

	// Create a buffer to be used for converting ARGB to tensor
	if (bDynamic)
		pBGRABuf = new unsigned char[outWidth*outHeight * 4];

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

	// If the width and height are not matched with the image width and height, scale the image
	if (!bDynamic && (outWidth != uiWidth || outHeight != uiHeight))
	{
		// 转化为Pre-multiplexed BGRA格式的WICBitmap
		if (FAILED(hr = m_spWICImageFactory->CreateBitmapFromSource(
			spConverter.Get(), WICBitmapCacheOnDemand, &spHandWrittenBitmap)))
			goto done;

		// 将转化为Pre-multiplexed BGRA格式的WICBitmap的原始图片转换到D2D1Bitmap对象中来，以便后面的缩放处理
		if (FAILED(hr = spRenderTarget->CreateBitmapFromWicBitmap(spHandWrittenBitmap.Get(), &spD2D1Bitmap)))
			goto done;

		// 将图片进行缩放处理，转化为m_outWidthxm_outHeight的图片
		spRenderTarget->BeginDraw();

		spRenderTarget->FillRectangle(dst_rect, spBGBrush.Get());

		if (GetImageDrawRect(outWidth, outHeight, uiWidth, uiHeight, dst_rect))
			spRenderTarget->DrawBitmap(spD2D1Bitmap.Get(), &dst_rect);

		spRenderTarget->EndDraw();

		//ImageProcess::SaveAs(spNetInputBitmap, L"I:\\test.png");

		// 并将图像每个channel中数据转化为[-1.0, 1.0]的raw data
		hr = spNetInputBitmap->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);
	}
	else
		hr = spConverter->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);

	float* res_data = (float*)malloc(3 * outWidth * outHeight * sizeof(float));
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < outHeight; i++)
		{
			for (int j = 0; j < outWidth; j++)
			{
				int pos = c * outWidth*outHeight + i * outWidth + j;
				res_data[pos] = ((pBGRABuf[i * outWidth * 4 + j * 4 + 2 - c]) / 255.0f - m_RGB_means[c]) / m_RGB_stds[c];
			}
		}
	}

	tensor = torch::from_blob(res_data, { 1, 3, outWidth, outHeight }, FreeBlob);

	hr = S_OK;

done:
	if (wszInputFile != NULL && wszInputFile != cszImageFile)
		delete[] wszInputFile;

	if (pBGRABuf != m_pBGRABuf)
		delete[] pBGRABuf;

	return hr;
}

HRESULT ImageProcess::ToTensor(std::vector<tstring> strImageFiles, torch::Tensor& tensor)
{
	HRESULT hr = S_OK;
	ComPtr<IWICBitmapDecoder> spDecoder;				// Image decoder
	ComPtr<IWICBitmapFrameDecode> spBitmapFrameDecode;	// Decoded image
	ComPtr<IWICBitmapSource> spConverter;				// Converted image
	ComPtr<IWICBitmap> spHandWrittenBitmap;				// The original bitmap
	ComPtr<ID2D1Bitmap> spD2D1Bitmap;					// D2D1 bitmap

	ComPtr<IWICBitmap> spNetInputBitmap = m_spNetInputBitmap;
	ComPtr<ID2D1RenderTarget> spRenderTarget = m_spRenderTarget;
	ComPtr<ID2D1SolidColorBrush> spBGBrush = m_spBGBrush;

	UINT uiFrameCount = 0;
	UINT uiWidth = 0, uiHeight = 0;
	UINT outWidth = m_outWidth;
	UINT outHeight = m_outHeight;
	WICPixelFormatGUID pixelFormat;
	unsigned char* pBGRABuf = m_pBGRABuf;
	D2D1_RECT_F dst_rect = { 0, 0, outWidth, outHeight };
	WICRect rect = { 0, 0, outWidth, outHeight };
	wchar_t wszImageFile[MAX_PATH + 1] = { 0 };
	const wchar_t* wszInputFile = NULL;
	float* res_data = NULL;

	static long long toTensorDuration = 0LL;

	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	if (strImageFiles.size() == 0)
		return E_INVALIDARG;

	if (outWidth != 0 && outHeight != 0)
		res_data = new float[strImageFiles.size() * 3 * outWidth*outHeight];

	for (size_t b = 0; b < strImageFiles.size(); b++)
	{
		int pos = 0;
		BOOL bDynamic = FALSE;
#ifndef _UNICODE
		if (MultiByteToWideChar(CP_UTF8, 0, strImageFiles[b].c_str(), -1, wszImageFile, MAX_PATH + 1) == 0)
		{
			hr = E_FAIL;
			goto done;
		}
		wszInputFile = (const wchar_t*)wszImageFile;
#else
		wszInputFile = strImageFiles[b].c_str();
#endif

		// 加载图片, 并为其创建图像解码器
		if (FAILED(m_spWICImageFactory->CreateDecoderFromFilename(wszInputFile, NULL,
			GENERIC_READ, WICDecodeMetadataCacheOnDemand, &spDecoder)))
			goto done;

		// 得到多少帧图像在图片文件中，如果无可解帧，结束此函数
		if (FAILED(hr = spDecoder->GetFrameCount(&uiFrameCount)) || uiFrameCount == 0)
			goto done;

		// 得到第一帧图片
		if (FAILED(hr = hr = spDecoder->GetFrame(0, &spBitmapFrameDecode)))
			goto done;

		// 得到图片大小
		if (FAILED(hr = spBitmapFrameDecode->GetSize(&uiWidth, &uiHeight)))
			goto done;

		// 调整转换和输出
		if (outWidth == 0)
		{
			outWidth = uiWidth;
			dst_rect.right = uiWidth;
			rect.Width = uiWidth;
			bDynamic = TRUE;
		}

		if (outHeight == 0)
		{
			outHeight = uiHeight;
			dst_rect.bottom = uiHeight;
			rect.Height = uiHeight;
			bDynamic = TRUE;
		}

		// Create a buffer to be used for converting ARGB to tensor
		if (bDynamic)
		{
			if (pBGRABuf == NULL)
				pBGRABuf = new unsigned char[outWidth*outHeight * 4];

			if (res_data == NULL)
				res_data = new float[strImageFiles.size() * 3 * outWidth*outHeight];
		}

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

		// If the width and height are not matched with the image width and height, scale the image
		if (!bDynamic && (outWidth != uiWidth || outHeight != uiHeight))
		{
			// 转化为Pre-multiplexed BGRA格式的WICBitmap
			if (FAILED(hr = m_spWICImageFactory->CreateBitmapFromSource(
				spConverter.Get(), WICBitmapCacheOnDemand, &spHandWrittenBitmap)))
				goto done;

			// 将转化为Pre-multiplexed BGRA格式的WICBitmap的原始图片转换到D2D1Bitmap对象中来，以便后面的缩放处理
			if (FAILED(hr = spRenderTarget->CreateBitmapFromWicBitmap(spHandWrittenBitmap.Get(), &spD2D1Bitmap)))
				goto done;

			// 将图片进行缩放处理，转化为m_outWidthxm_outHeight的图片
			spRenderTarget->BeginDraw();

			spRenderTarget->FillRectangle(dst_rect, spBGBrush.Get());

			if (GetImageDrawRect(outWidth, outHeight, uiWidth, uiHeight, dst_rect))
				spRenderTarget->DrawBitmap(spD2D1Bitmap.Get(), &dst_rect);

			spRenderTarget->EndDraw();

			//ImageProcess::SaveAs(spNetInputBitmap, L"I:\\test.png");

			// 并将图像每个channel中数据转化为[-1.0, 1.0]的raw data
			hr = spNetInputBitmap->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);
		}
		else
			hr = spConverter->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);

		pos = b * 3 * outWidth*outHeight;
		for (int c = 0; c < 3; c++)
		{
			for (int i = 0; i < outHeight; i++)
			{
				for (int j = 0; j < outWidth; j++)
				{
					int cpos = pos + c * outWidth*outHeight + i * outWidth + j;
					res_data[cpos] = ((pBGRABuf[i * outWidth * 4 + j * 4 + 2 - c]) / 255.0f - m_RGB_means[c]) / m_RGB_stds[c];
				}
			}
		}
	}

	tensor = torch::from_blob(res_data, { (long long)strImageFiles.size(), 3, outWidth, outHeight }, FreeBlob);

	hr = S_OK;

done:
	if (pBGRABuf != m_pBGRABuf)
		delete[] pBGRABuf;

	tm_end = std::chrono::system_clock::now();
	toTensorDuration += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();

	//printf("Load batch tensors, cost %lldh:%02dm:%02d.%03ds\n", 
	//	toTensorDuration/1000/3600, 
	//	(int)(toTensorDuration/1000/60%60), 
	//	(int)(toTensorDuration/1000%60),
	//	(int)(toTensorDuration%1000));

	return hr;
}

void ImageProcess::SaveAs(ComPtr<IWICBitmap>& bitmap, PCWSTR filename)
{
	HRESULT hr = S_OK;
	GUID guid = GUID_ContainerFormatPng;
	ComPtr<IWICImagingFactory> spWICImageFactory;

	PCWSTR cwszExt = wcsrchr(filename, L'.');
	if (cwszExt != NULL)
	{
		if (_wcsicmp(cwszExt, L".png") == 0)
			guid = GUID_ContainerFormatPng;
		else if (_wcsicmp(cwszExt, L".jpg") == 0 || _wcsicmp(cwszExt, L".jpeg") == 0 || _wcsicmp(cwszExt, L".jpg+") == 0)
			guid = GUID_ContainerFormatJpeg;
	}

	ComPtr<IStream> file;
	GUID pixelFormat;
	ComPtr<IWICBitmapFrameEncode> frame;
	ComPtr<IPropertyBag2> properties;
	ComPtr<IWICBitmapEncoder> encoder;
	UINT width, height;

	hr = SHCreateStreamOnFileEx(filename,
		STGM_CREATE | STGM_WRITE | STGM_SHARE_EXCLUSIVE,
		FILE_ATTRIBUTE_NORMAL,
		TRUE, // create
		nullptr, // template
		file.GetAddressOf());
	if (FAILED(hr))
		goto done;

	if (FAILED(CoCreateInstance(CLSID_WICImagingFactory,
		nullptr,
		CLSCTX_INPROC_SERVER,
		IID_IWICImagingFactory,
		(LPVOID*)&spWICImageFactory)))
		goto done;

	hr = spWICImageFactory->CreateEncoder(guid,
		nullptr, // vendor
		encoder.GetAddressOf());
	if (FAILED(hr))
		goto done;

	hr = encoder->Initialize(file.Get(), WICBitmapEncoderNoCache);
	if (FAILED(hr))
		goto done;

	hr = encoder->CreateNewFrame(frame.GetAddressOf(), properties.GetAddressOf());
	if (FAILED(hr))
		goto done;

	if (FAILED(hr = frame->Initialize(properties.Get())))
		goto done;

	if (FAILED(hr = bitmap->GetSize(&width, &height)))
		goto done;

	if (FAILED(hr = frame->SetSize(width, height)))
		goto done;

	if (FAILED(hr = bitmap->GetPixelFormat(&pixelFormat)))
		goto done;

	{
		auto negotiated = pixelFormat;
		if (FAILED(hr = frame->SetPixelFormat(&negotiated)))
			goto done;
	}

	if (FAILED(hr = frame->WriteSource(bitmap.Get(), nullptr)))
		goto done;

	if (FAILED(hr = frame->Commit()))
		goto done;

	if (FAILED(hr = encoder->Commit()))
		goto done;

done:
	return;
}
