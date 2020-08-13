#include <torch/torch.h>

#define NOMINMAX

#include <wincodec.h>
#include <wincodecsdk.h>
#include <wrl/client.h>
#include <d3d.h>
#include <d2d1.h>
#include <d2d1_2.h>
#include <shlwapi.h>
#include "ImageProcess.h"

using namespace Microsoft::WRL;
using namespace torch::nn;

#pragma once

#define MAX_LABEL_NAME		2048

class VGGNet : public Module
{
public:
	using tstring = std::basic_string<TCHAR, std::char_traits<TCHAR>, std::allocator<TCHAR>>;

	VGGNet(int num_classes);
	~VGGNet();

	torch::Tensor	forward(torch::Tensor& input);
	int				train(const TCHAR* szTrainSetRootPath, const TCHAR* szTrainSetStateFilePath);
	void			verify(const TCHAR* szTrainSetRootPath, const TCHAR* szTrainSetStateFilePath);
	int				savenet(const TCHAR* szTrainSetStateFilePath);
	int				loadnet(const TCHAR* szTrainSetStateFilePath);
	void			classify(const TCHAR* szImageFile);

public:
	int64_t			num_flat_features(torch::Tensor input);
	HRESULT			loadImageSet(const TCHAR* szImageSetRootPath,
								 std::vector<tstring>& image_files,
								 std::vector<tstring>& image_labels,
								 std::vector<size_t>& image_shuffle_set,
								 bool bTrainSet = true, bool bShuffle = true);
	HRESULT			loadLabels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels);

protected:
	// block 1
	Conv2d			C1;
	BatchNorm2d		C1B;
	Conv2d			C3;
	BatchNorm2d		C3B;

	// block 2
	Conv2d			C6;
	BatchNorm2d		C6B;
	Conv2d			C8;
	BatchNorm2d		C8B;

	// block 3
	Conv2d			C11;
	BatchNorm2d		C11B;
	Conv2d			C13;
	BatchNorm2d		C13B;
	Conv2d			C15;
	BatchNorm2d		C15B;

	// block 4
	Conv2d			C18;
	BatchNorm2d		C18B;
	Conv2d			C20;
	BatchNorm2d		C20B;
	Conv2d			C22;
	BatchNorm2d		C22B;

	// block 5
	Conv2d			C25;
	BatchNorm2d		C25B;
	Conv2d			C27;
	BatchNorm2d		C27B;
	Conv2d			C29;
	BatchNorm2d		C29B;

	Linear			FC32;
	Linear			FC35;
	Linear			FC38;

	std::vector<tstring>
					m_image_labels;				// the image labels for this network
	ImageProcess	m_imageprocessor;
	bool			m_bEnableBatchNorm = true;
};

