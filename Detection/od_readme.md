# Object detection task in MMAU datasets

In Object Detection Benchmark,we use the cocodataset style to organize our data, the data we used in the paper's detection benchmark in the link.

## Data download
All the results were conducted by [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMYOLO](https://github.com/open-mmlab/mmyolo) toolbox, checkpoints and config file will be released sonn.

---

The Origin annotation files can be downloaded from here:

Refied version: [labels_6fps_GT](https://pan.baidu.com/s/1su8pcIx7GLvCD1qErkJWww?pwd=zfwz)

Raw version: [labels_30fps_diffdet_inference](https://pan.baidu.com/s/1ksUAbb0tdpOSKP87tpYUyA?pwd=gwl7)


*Note*: The refined version is a manually corrected labeled file with a frame rate of 6fps. This is what was used in the paper, and the cocodataset style dataset was obtained from this. raw version is obtained by inference from the [Diffusiondet](https://github.com/ShoufaChen/DiffusionDet) model trained on the refined version data, and raw version is the result of the inference and has not been processed in any way.

---

Cocodataset style labels and images files can be downloaded from here:

**[BaiDuNetDisk]** The data that we used in paper:[images](https://pan.baidu.com/s/1DQ8wlfte_JcC6MWhAsFZrw?pwd=fvpk), [labels](https://pan.baidu.com/s/1aoca1jCbZf_NErtibY6H7A?pwd=5icc).

**[BaiDuNetDisk]** MMAU-Detectv1(Reorganized data,also in cocodataset style): [images](https://pan.baidu.com/s/1bL4-ZFWcw3B28gBOrjwi7w?pwd=umdw), [labels](https://pan.baidu.com/s/1l777BoN2_z7vvLbqqHYqxA?pwd=y5y1)

All the download link of **google drive** will be uploded as soon as possible.

*Note* :The object detection data used in the paper and the improved version MMAU-Detectv1 differ in both file names and number of videos due to different data cleaning methods and organization, but both maintain the same cocodataset style and the same dataset division strategy. The dataset used in the paper is provided to ensure the reproducibility of our paper, while the organization of MMAU-Detectv1 allows for better access to the video and image metadata when needed.

## Installation

step1:
```
git clone https://github.com/jeffreychou777/LOTVS-MM-AU.git
cd LOTVS-MM-AU/Detection/mmdetection
conda create -n mmaudet python=3.8 -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
step2:
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.1

pip install -v -e .
```

## Quick Run

configs are stored in Detection/mmdetection/configs_mmau.

All uses of the config file are as described in the [mmdetection documentation](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html).
```
python tools/train.py config_path
bash tools/dist_train.sh config_path num_gpus
```