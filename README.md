# LOTVS-MMAU(Multi-Modal Accident video Understanding)

[project home page](http://www.lotvsmmau.net/)

## MM-AU Datasets

This is the official Repo for paper "Abductive Ego-View Accident Video Understanding for Safe Driving Perception"[CVPR2024 Highlight] [paper](https://arxiv.org/abs/2403.00436)

### Intorduction

MM-AU consists of two data sets, [MM-Cap](https://github.com/JWFangit/LOTVS-CAP) and [MM-DADA](https://github.com/JWFangit/LOTVS-DADA) together

### video_metadata annotations

An example:
'''
{
"video_hashcode": {
        "video_name": "1_1",
        "id": "1",
        "type": "1",
        "weather": "1",
        "light": "1",
        "scenes": "4",
        "linear": "1",
        "accident occurred": "1",
        "abnormal_start_frame": "30",
        "abnormal_end_frame": "115",
        "accident_frame": "63",
        "total_frames": "440",
        "t_ai": "30",
        "t_co": "63",
        "t_ae": "115",
        "texts": "a pedestrian crosses the road",
        "causes": "Pedestrian does not notice the coming vehicles when crossing the street",
        "measures": "When passing the zebra crossing, drivers must slow down. When pedestrians or non-motor vehicles cross the zebra crossing, they should stop and give way to other normal running vehicles; When crossing the road, pedestrians must follow the zebra crossing, carefully observe the traffic, and do not cross the road in a hurry."
    }
}
'''

## Task

MM-AU supports a variety of tasks due to its multimodal characteristics, and the following describes the application of MM-AU to various tasks.

### Object Detection

In Object Detection Benchmark,we use the Cocodataset format to organize our data, the data we used in the paper's detection benchmark in the link.
All the results were conducted by [MMDetection](https://github.com/open-mmlab/mmdetection) toolbox, checkpoints and config file will be released sonn.

Cocodataset format download[link](https://github.com/jeffreychou777)

## Citation

If our work and repo is helpful to you, please give us a **free star** and **cite our paper**!

---
'''
@article{fang2024abductive,
  title={Abductive Ego-View Accident Video Understanding for Safe Driving Perception},
  author={Fang, Jianwu and Li, Lei-lei and Zhou, Junfei and Xiao, Junbin and Yu, Hongkai and Lv, Chen and Xue, Jianru and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2403.00436},
  year={2024}
}
'''
