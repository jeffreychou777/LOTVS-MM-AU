# Object detection task in MMAU datasets

In Object Detection Benchmark,we use the cocodataset style to organize our data, the data we used in the paper's detection benchmark in the link.

## Data download
All the results were conducted by [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMYOLO](https://github.com/open-mmlab/mmyolo) toolbox, checkpoints and config file will be released sonn.

---

The Origin annotation files can be downloaded from here: -->

Refied version: [labels_6fps_GT](https://pan.baidu.com/s/16XMDxT4mr8oFHTwBXEuzOw?pwd=eujv)

Raw version: [labels_30fps_diffdet_inference](https://pan.baidu.com/s/1wF-4tjXeXl1QVHINAmgwqA?pwd=ggyn)

The MM-AU datasets have a refined version and a raw version labels respectively.(The download link will be upload soon)

*Note*: The refined version is a manually corrected labeled file with a frame rate of 6fps. This is what was used in the paper, and the cocodataset style dataset was obtained from this. raw version is obtained by inference from the [Diffusiondet](https://github.com/ShoufaChen/DiffusionDet) model trained on the refined version data, and raw version is the result of the inference and has not been processed in any way.

---

Cocodataset style labels and images files can be downloaded from here:

**[BaiDuNetDisk]** The data that we used in paper:[Link](https://pan.baidu.com/s/1jqGiH8r2TrLxfSoy5k9uiQ?pwd=f8gj).

**[BaiDuNetDisk]** MMAU-Detectv1(Reorganized data,also in cocodataset style): [Link](https://pan.baidu.com/s/1MLHgaBgaBgzNnci5vNJCXw?pwd=e38j)

All the format of dataset mentioned above can be downloaded in the **[Huggingface]** repo: [Link](https://huggingface.co/datasets/JeffreyChou/MM-AU/tree/main)

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
