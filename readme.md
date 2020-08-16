# VGG net

## How to set up the libtorch environment based on Visual Studio

https://blog.csdn.net/defi_wang/article/details/107450428

## Code Introduce

https://blog.csdn.net/defi_wang/article/details/107889818
https://blog.csdn.net/defi_wang/article/details/108032208

## Convert Image to Tensor

https://blog.csdn.net/defi_wang/article/details/107936757

## How to run it?
*VGGNet [options] command [arg...]*

### *Commands*

|command|description|
|--------------|-----------------|
|state | Show the VGG net state |
|train | train the network |
|verify | verify the pre-trained network with the test-set|
|classify | classify an input picture with the pre-trained network |

### *optoins*
|option|description|
|------|--------|
|-v|Verbose mode to output more logs|
|-y | Proceed the operation without any prompt |

## *arguments for command*
### **state**
*VGGNet state*

### **train**
*VGGNet train image_set_root_path train_output [-b/--batchsize batchsize] [-e/--epochnum epochnum] [-l/--learningrate fixed_learningrate] [-bn/-batchnorm] [-n numclass] [-s/--smallimagesize] [--showloss once_num_batch] [--clean]*

### #*args*
|name|shortname|arg|description|
|----|---------|---|-----------|
|**batchsize**|**b**|batchsize|the batch size of sending to network|
|**epochnum**|**e**|epochnum|the number of train epochs|
|**learningrate**|**l**|learing rate|the fixed learning rate<br>(\*)if it is not specified, default learning rate is used, dynamic learning rate is used|
|**batchnorm**<br>**bn**|*n/a*|*n/a*|enable batchnorm after CNN |
|**numclass**|**n**|num of classes|The specified final number of classes, the default value is 1000|
|**smallsize**|**s**|*n/a*|Use 32x32 input instead of the original 224\*224|
|**showloss**|*n/a*|once_num_batch|stat. the loss every num batch |
|**clean**|*n/a*|*n/a*|clean the previous pre-trained net state file |

Train a network with the specified train image set, and the image set folder structure is like as

```
{image_set_root_path} 
  |-training_set
  |   |-tag1
  |   |  |-- images......
  |   |-tag2
  |   |  |-- images......
  |   ......
  |_test_set
      |-tag1
      |  |-- images......
```

### **verify**
*VGGNet verify image_set_root_path pretrain_network*
Verify the test-set and show the pass-rate and other information

```
VGGNet verify I:\CatsDogs I:\catdog.pt
```

### **classify**
*VGGNet classify pretrain_network image_file_path*
With the specified pre-trained network, classify a image.
```
VGGNet classify I:\catdog.pt PIC_001.png
```