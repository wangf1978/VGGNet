#include <torch/torch.h>

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
	ImageProcess();
	~ImageProcess();

	HRESULT			Init(UINT outWidth, UINT outHeight);
	HRESULT			ToTensor(const TCHAR* cszImageFile, torch::Tensor& tensor, float med=.0f, float std=1.f);
	void			Uninit();

	static void		SaveAs(ComPtr<IWICBitmap>& bitmap, PCWSTR filename);

protected:
	static bool		GetImageDrawRect(UINT target_width, UINT target_height, UINT image_width, UINT image_height, D2D1_RECT_F& dst_rect);

protected:
	ComPtr<ID2D1Factory>	
					m_spD2D1Factory;			// D2D1 factory
	ComPtr<IWICImagingFactory>	
					m_spWICImageFactory;		// Image codec factory
	ComPtr<IWICBitmap>		
					m_spNetInputBitmap;			// The final bitmap 1x224x224
	ComPtr<ID2D1RenderTarget>
					m_spRenderTarget;			// Render target to scale image
	ComPtr<ID2D1SolidColorBrush>
					m_spBGBrush;				// the background brush
	unsigned char*	m_pBGRABuf = NULL;

	UINT			m_outWidth = 0;
	UINT			m_outHeight = 0;
};

