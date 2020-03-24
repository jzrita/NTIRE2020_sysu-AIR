# A-Fast-Feedback-Network-for-Large-Scale-Image-SR
FFNSR @ NTIRE 2020

This repository is Pytorch code for our proposed FFNSR.

The code is developed by team [sysu-AIR](https://github.com/jzrita/NTIRE2020_sysu-AIR), and tested on Ubuntu 16.04 environment (Python 3.6, PyTorch 1.0.1, CUDA 10.0) with 2080Ti GPUs.The details about our proposed FFNSR can be found in [our main paper].

# Contents
1. [FFNSR](#FFNSR)
2. [Requirements](#Requirements)
3. [Test](#Test)
4. [Train](#Train)
5. [Results](#results)


## FFNSR

这里再填一些介绍性的文字或者图片。

* Number of parameters:  xxx,xxx,xxx ()

* Average PSNR on validation data: xxxx dB

* Average inference time (RTX 2080 Ti) on validation data: xxxx second 

    Note: We selected the best average inference time among three trials

## Requirements
- Python 3 (Anaconda is recommended)
- skimage
- imageio
- Pytorch (Pytorch version >=0.4.1 is recommended)
- tqdm 
- pandas
- cv2 (pip install opencv-python)
- Matlab 

## Test

#### Quick start
1. Clone this repository:

   ```shell
   git clone https://github.com/sysu17364012/A-Fast-Feedback-Network-for-Large-Scale-Image-SR
   ```

2. Download our pre-trained model from the links below, unzip the models and place them to `./models`.

    [Click_here_to_download](https://www.baidu.com/)
    
 
3. CD the folder and install the requirements:

   ```shell
   cd A-Fast-Feedback-Network-for-Large-Scale-Image-SR && pip install -r requirements.txt
   ```

4. Place the LR pictures to `./picture`.

   ```shell
   ./picture/1601.png
   ./picture/1602.png
   ./picture/1603.png
   ...
   ```

5. Then run the **following commands** to test the model:

   ```shell
    python test.py
   ```

## Train
1. Run command to train the model：
   ```shell
   cd A-Fast-Feedback-Network-for-Large-Scale-Image-SR
   python train.py
   ```

## Results
主要放图片或者psnr评分
