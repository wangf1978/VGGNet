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

### *options*
|option|description|
|------|--------|
|-v |Verbose mode to output more logs|
|-y | Proceed the operation without any prompt |

## *arguments for command*
### **state**
*VGGNet state [--bn/-batchnorm] [-n numclass] [-s/--smallsize] [train_output]*

If no arg is specified, it will print the VGG-D net at default.

examples:
```
VGGNet.exe state --bn --numclass 10 --smallsize
```
print the neutral network state with batchnorm layers, the output number of classes and use the 32x32 small input image instead the 224x224 image.
```
VGGNet.exe I:\catdog.pt
```
print the information of neutral network loading from I:\catdog.pt.

#### *args*
|name|shortname|arg|description|
|----|---------|---|-----------|
|**batchnorm**<br>**bn**|*n/a*|*n/a*|enable batchnorm after CNN |
|**numclass**|**n**|num of classes|The specified final number of classes, the default value is 1000|
|**smallsize**|**s**|*n/a*|Use 32x32 input instead of the original 224\*224|


### **train**
*VGGNet train image_set_root_path train_output [-b/--batchsize batchsize] [-e/--epochnum epochnum] [-l/--learningrate fixed_learningrate] [--bn/--batchnorm] [-n numclass] [-s/--smallsize] [--showloss once_num_batch] [--clean]*

#### *args*
|name|shortname|arg|description|
|----|---------|---|-----------|
|**batchsize**|**b**|batchsize|the batch size of sending to network|
|**epochnum**|**e**|epochnum|the number of train epochs|
|**learningrate**|**l**|learning rate|the fixed learning rate<br>(\*)if it is not specified, default learning rate is used, dynamic learning rate is used|
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
Examples
```
VGGNet.exe train I:\CatDog I:\catdog.pt --bn -b 64 -l 0.0001 --showloss 10
```
Train the image set lies at I:\CatDog, and save the output to I:\catdog.pt, the batchnorm layers will be introduced, and the batch size is 64, the learning rate use the fixed 0.0001, and show the loss rate every 10 batches.
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