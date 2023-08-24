# LongShortNet: Exploring Temporal and Semantic Features Fusion in Streaming Perception

In the sphere of autonomous driving, streaming perception plays a pivotal role. It's vital to achieve a fine balance between the system's latency and accuracy. **LongShortNet** emerges as an innovative model, intertwining both long-term temporal dynamics and short-term spatial semantics, hence fostering enhanced real-time perception. This fusion leads to a model that promises greater efficacy in complex autonomous driving scenarios.

<p align="center">
  <img src="https://github.com/zhiqic/LongShortNet/assets/65300431/8f1a6ec4-10a6-4bbc-967c-e660110419ac" alt="Figure 1" width="555"/>
  <br>
  <i>Fig. 1: (a) Offline detection (VOD) vs. streaming perception. Streaming perception operates in real-time, adapting swiftly to motion changes. (b) A timeline depicting processing time.</i>
  <br><br>
  <img src="https://github.com/zhiqic/LongShortNet/assets/65300431/897bcfa0-ad68-4947-9655-1e1529725cd4" alt="Figure 2" width="583"/>
  <br>
  <i>Fig. 2: (a) StreamYOLO's performance comparison, with red and orange boxes denoting ground truth and predictions respectively. (b) sAP comparison between LongShortNet and StreamYOLO. For more visuals, visit [here](https://rebrand.ly/wgtcloo).</i>
</p>


## Methodology

LongShortNet embarks on a unique fusion strategy, the **Long Short Fusion Module(LSFM)**, which is the crux of our architecture. LSFM employs multiple fusion schemes to ensure the effective amalgamation of features, providing the network the ability to react dynamically to real-time changes. The essence of LongShortNet is its capability to seamlessly combine temporal and semantic features. The methodology is split into two main components as illustrated below:

<p align="center">
  <img src="https://github.com/zhiqic/LongShortNet/assets/65300431/4ceb546e-2daa-4588-b269-3df084eb2f39" alt="Figure 3a" width="774"/>
  <br>
  <i>Fig. 3(a): Overview of the LongShortNet framework.</i>
  <i>Fig. 3(b): A detailed view of the fusion schemes in LSFM (LongShort Feature Module).</i>
  <br>
</p>


For a more comprehensive understanding of the approach and its benefits, please take a look at our [ICASSP paper](https://arxiv.org/abs/2210.15518).


## Benchmark

<center>

| Model | Size | Velocity | sAP<br>0.5:0.95 | sAP50 | sAP75 | Weights |
|:------:|:----:|:--------:|:---------------:|:-----:|:-----:|:-------:|
|[LongShortNet-S](./cfgs/longshortnet/s_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8.py) | 600×960 | 1x | 29.8 | 50.4 | 29.5 | [link](https://drive.google.com/file/d/13ESdjetcccOKnU0fg54b6czuxBH76C_7/view?usp=share_link) |
|[LongShortNet-M](./cfgs/longshortnet/m_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8.py) | 600×960 | 1x | 34.1 | 54.8 | 34.6 | [link](https://drive.google.com/file/d/1AFzD2bTSTtuCCWBk2AnU1t9uHVGD1cM_/view?usp=share_link) |
|[LongShortNet-L](./cfgs/longshortnet/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8.py) | 600×960 | 1x | 37.1 | 57.8 | 37.7 | [link](https://drive.google.com/file/d/15D6VL_QcL1qBYjBmZCAEa0PNp0TM67vg/view?usp=share_link) |
|[LongShortNet-L](./cfgs/longshortnet/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8_1200x1920.py) | 1200×1920 | 1x | **42.7** | **65.4** | **45.0** | [link](https://drive.google.com/file/d/1gI5a2Pf1MOnCkxeNIHLnbgesEOZEXER2/view?usp=share_link) |

</center>


## Quick Start

### Installation
You can refer to [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO) to install the whole environments.

### Train
We use COCO models offered by [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO) as our pretrained models.
```shell
bash run_train.sh
```

### Evaluation
```shell
bash run_eval.sh
```

## Citation
Please cite the following paper if this repo helps your research:
```bibtex
@inproceedings{li2023longshortnet,
  title={Longshortnet: Exploring temporal and semantic features fusion in streaming perception},
  author={Li, Chenyang and Cheng, Zhi-Qi and He, Jun-Yan and Li, Pengyu and Luo, Bin and Chen, Hanyuan and Geng, Yifeng and Lan, Jin-Peng and Xie, Xuansong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgment
A major part of this project's foundation is built on [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO). We extend our gratitude to the authors. We are also grateful to Alibaba Group's DAMO Academy for their invaluable support.

## License
LongShortNet is under the Apache 2.0 license. Refer to the LICENSE file for more details.
