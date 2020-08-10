#define NOMINMAX

#include <wincodec.h>
#include <wincodecsdk.h>
#include <wrl/client.h>
#include <d3d.h>
#include <d2d1.h>
#include <d2d1_2.h>
#include <shlwapi.h>

#pragma once

using namespace Microsoft::WRL;

class ImageProcess
{
public:
	static void		SaveAs(ComPtr<IWICBitmap>& bitmap, PCWSTR filename);
};

