#include "ImageProcess.h"

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
