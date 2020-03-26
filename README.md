# A-Fast-Feedback-Network-for-Large-Scale-Image-SR
FFNSR @ NTIRE 2020

This repository is Pytorch code for our proposed FFNSR.

The code is developed by team [sysu-AIR](https://github.com/jzrita/NTIRE2020_sysu-AIR), and tested on Ubuntu 16.04 environment (Python 3.6, PyTorch 1.0.1, CUDA 10.0) with 2080Ti GPUs.The details about our proposed FFNSR can be found in [our main paper].

# Contents
1. [FFNSR](#FFNSR)
2. [Requirements](#Requirements)
3. [Test](#Test)
4. [Train](#Train)
5. [Result](#Result)


## FFNSR

The **sysu-AIR** team proposed **A Fast Feedback Network for Large Scale Image Super-Resolution**. Inspired by **SRFBN** and **IMDN**, the proposed FFNSR is still reserved the RNN structure but with a information multi-distillation module (IMDM), which can beneﬁt image SR tasks and accelerate inference speed.

* Number of parameters:  2,099,625

* Average PSNR on validation data: 29.01096 dB

* Average inference time (RTX 2080 Ti) on validation data: 4.35 second 

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

    [Click_here_to_download](https://pan.baidu.com/s/1XytyS1XyUidfP8uDeBuN6g)
    (code: a3mu)
    
 
3. CD the folder and install the requirements:

   ```shell
   cd NTIRE2020_sysu-AIR && pip install -r requirements.txt
   ```

4. Place the LR pictures to `./picture`.

   ```shell
   ./picture/1601.png
   ./picture/1602.png
   ./picture/1603.png
   ...
   ```

5. Edit `./options/test/test_SRFBN_example.json` for your needs according to [`./options/test/README.md`.](./options/test/README.md)

6. Then run the **following commands** to test the model:

   ```shell
    python test.py
   ```

## Train
1. Edit `./options/train/train_SRFBN.json` for your needs according to [`./options/train/README.md`.](./options/train/README.md)

2. Run command to train the model：
   ```shell
   cd NTIRE2020_sysu-AIR
   python train.py
   ```

3. You can monitor the training process in `./experiments`.

## Result
1. Download and view our test result.

    [Click_here_to_download](https://pan.baidu.com/s/12zH-7AssJd3HJql-gjrrlw)
    (code: fiz9)
