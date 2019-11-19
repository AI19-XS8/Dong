

[助教推荐的 pytorch 视频](https://morvanzhou.github.io/tutorials/machine-learning/torch/1-1-D-feature-representation/)






[Adversarial Examples that Fool both ComputerVision and Time-Limited Humans](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fpapers.nips.cc%2Fpaper%2F7647-adversarial-examples-that-fool-both-computer-vision-and-time-limited-humans.pdf)

[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)  

GAN

[NIPS Adversarial 搜索结果](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=nips+Adversarial&btnG=)




[](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_cvpr_2018%2Fpapers%2FLiao_Defense_Against_Adversarial_CVPR_2018_paper.pdf)



# 正文开始

# Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser
https://arxiv.org/abs/1712.02976


## Fangzhou Liao∗, Ming Liang∗, Yinpeng Dong, Tianyu Pang, Xiaolin Hu†, Jun Zhu
Department of Computer Science and Technology,  
Tsinghua Lab of Brain and Intelligence,  
Beijing National Research Center for Information Science and Technology,    
BNRist Lab  
Tsinghua University, 100084 China



## Content
## Abstract

Instead of using apixel-level reconstruction loss function as standard denoisers, we set the loss function as the difference between top-level outputs of the target model induced by original and adversarial examples (Figure 1). 
We name the denoiser trainedby this loss function “high-level representation guided de-noiser” (HGD)




## Notations
$x$ : the clean image from agiven dataset  

$y$ : the class.   

$y_{true}$ : the ground truth label
 
$f: x \to y$ : A neural network, the target model

$f_{l}(x)$ : the feature vector at layer $l$ of an input $x$ 

$p(y|x)$ : its predicted probability of class $y$  
$y_{x}= arg max_{y} p(y|x)$ : the predicted class of $x$

$J(x, y)$ : the loss function of the classifier giventhe input $x$ and its target class $y$, the cross-entropy loss for image classification

$x^{*}$ : the adversarial example generated from $x$  

$\epsilon$ : the magnitude of adversarial perturbation, measured by acertain distance metric

# Existing methods for adversarial attacks

## L-BFGS
Szegedy et al.  
a box-constrained L-BFGS al-gorithm to generate targeted adversarial examples, whichbias the predictions to a specified classytarget.  

More specifically, they minimize the weighted sum of $\epsilon$ and $J(x, y_{target})$ while constraining the elements of $x^{*}$ to benormal pixel value.


##  Fast Gradient Sign Method (FGSM)
Goodfellow et al. 

$$
x^{*}=x+\epsilon·sign(∇_{x}J(x, y)) 
$$

### modified FGSM

$$
x^{*}=x-\epsilon·sign(∇_{x}J(x, y_{target}))
$$

## IFGSM
rakin et al.

an iter-ative FGSM (IFGSM) attack by repeating FGSM fornsteps(IFGSMn). 
IFSGM usually results in higher classificationerror than FGSM.


## 总结 & 过渡
As white-box at-tacks are less likely to happen in practical systems, defenses against black-box attacks are more desirable.


# Existing methods for defenses


## Adversarial training  
one of the most extensively investigated defenses against adversarial attacks.

It aims to train a robust model from scratch on a trainingset augmented with adversarially perturbed data.

缺点
1. 只能用在小的干净的数据集  
small image datasets it even improves the accuracy of clean images, although this effect is not found on ImageNet dataset. 

2. 耗时  
However, adversarial training is moretime consuming than training on clean images only, because online adversarial example generation needs extra computation, and it takes more epochs to fit adversarial examples. 

所以现在只用 FGSM

Practical adversarial training on the ImageNet dataset only adopts FGSM.




## Preprocessing based methods  
process the inputs with certain transformations to   
remove the adversarial noise    
then send these inputs to the target model.

### denoising auto-encoders
Gu and Rigazio  

the use of denoising auto-encoders as a defense.


### filters
Osadchy et al.   
a set of filters to remove the adversarial noise, such as the median filter, averaging filter and Gaussian low-pass filter 

### preprocessing transformations
Graese et al.   
assess the defending performance of a set of preprocessing transformations on MNIST digits, including the perturbations introduced by image acquisition process, fusion of crops and binarization. 


### JPEG
Das et al.   
preprocess images with JPEG compression to reduce the effect of adversarialnoises. 

### a two-step defense model
Meng and Chen   

detects the adversarial input 

then reform it based on the difference between the manifolds of clean and adversarial examples.


### 缺点
1. these methods are usually evaluated on small images
2. some method effective on small images may not transfer well to large images


## gradient masking effect 

These defenses apply some regularizers or smooth labels to make the model output less sensitive to the perturbation on input.



### the deep contrastive network
Gu and Rigazio

use a layer-wise contrastive penalty term to achieve output invariance to input perturbation. 


### knowledge distillation
Papernot et al. 

adapts knowledge distillation to adversarial defense,   
and uses the output of another model as soft labels to trainthe target model. 

### saturating networks
Nayebi and Surya

use saturating networks for robustness to adversarial noises. 

The loss function is designed to encourage the activations to be in their saturating regime.




### 没解决根本问题，只是让问题更复杂了

The basic problem with these gradient masking approaches is 
that they fail to solve the vulnerability of the models to adversarial attacks, but just make the construction of white-box adversarial examples more difficult. 
These defenses still suffer from black-box attacks generated on other models.


## Methods



### denoising autoencoder 

[DAE 简介](https://blog.csdn.net/n1007530194/article/details/78369429)

论文中关于Denoising Auto-encoder的示意图如下，其中x是原始的输入数据，Denoising Auto-encoder以一定概率把输入层节点的值置为0，从而得到含有噪音的模型输入xˆ。这和dropout很类似，不同的是dropout是隐含层中的神经元置为0。  
![DAE](https://img-blog.csdn.net/20171029173315272?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbjEwMDc1MzAxOTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
以这丢失的数据x’去计算y，计算z，并将z与原始x做误差迭代，这样，网络就学习了这个破损（原文叫Corruputed）的数据。  

Denoising Auto-encoder与人的感知机理类似，比如人眼看物体时，如果物体某一小部分被遮住了，人依然能够将其识别出来。
人在接收到多模态信息时（比如声音，图像等），少了其中某些模态的信息有时也不会造成太大影响。


### DUNET
DAE has a bottleneck structure between the encoder anddecoder.  

This bottleneck may constrain the transmissionof fine-scale information necessary for reconstructing high-resolution images. 

To overcome this problem, we modify DAE with the U-net structure and propose the denoising U-net (DUNET, see Figure 2 right). 

![figure2](https://github.com/lengyuner/MachineLearning/blob/master/DeepLearning/pic/DefenseAgainstAdversarialAttacksUsingHighLevelRepresentationGuidedDenoiserFigure2.png?raw=true)
#### 
DUNET is different from DAE in two aspects. 

1. First, similar to the Ladder network, DUNET adds lateral connections from encoder layers to their corresponding decoder layers in thesame scale. 
2. Second, the learning objective of DUNET is the adversarial noise ($d\hat{x}$ in Figure 2), instead of reconstructing the whole image as in DAE. This residual learning is implemented by the shortcut from input to output to additively combine them. The clean image can be readily ob-tained by subtracting the noise (adding $-d\hat{x}$) from the corrupted input.



### Network architecture


![figure3](https://github.com/lengyuner/MachineLearning/blob/master/DeepLearning/pic/DefenseAgainstAdversarialAttacksUsingHighLevelRepresentationGuidedDenoiserFigure3.png?raw=true)


$$
\hat{x}=x^{*}-d\hat{x}
$$


### PGD
Pixel guided denoiser

Loss Function

$$
L=||x-\hat{x}||
$$
### HGD
High-level representation guided denoiser


$$
L=||f_{l}(\hat{x})-f_{l}(x)||
$$
1. FGD
$l=-2$
2. LGD
$l=-1$














