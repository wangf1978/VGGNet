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

	torch::Tensor	forward(torch::Tensor input);
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
	Conv2d			C3;

	// block 2
	Conv2d			C6;
	Conv2d			C8;

	// block 3
	Conv2d			C11;
	Conv2d			C13;
	Conv2d			C15;

	// block 4
	Conv2d			C18;
	Conv2d			C20;
	Conv2d			C22;

	// block 5
	Conv2d			C25;
	Conv2d			C27;
	Conv2d			C29;

	Linear			FC32;
	Linear			FC35;
	Linear			FC38;

	std::vector<tstring>
					m_image_labels;				// the image labels for this network
	ImageProcess	m_imageprocessor;
};

