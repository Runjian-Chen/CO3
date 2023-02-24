# CO3: Cooperative Unsupervised 3D Representation Learning for Autonomous Driving

## Paper

Arxiv: https://arxiv.org/abs/2206.04028

If you are interested in our work and use the model or code, please consider cite:
      
      @inproceedings{
      chen2023co,
      title={{CO}3: Cooperative Unsupervised 3D Representation Learning for Autonomous Driving},
      author={Runjian Chen and Yao Mu and Runsen Xu and Wenqi Shao and Chenhan Jiang and Hang Xu and Yu Qiao and Zhenguo Li and Ping Luo},
      booktitle={The Eleventh International Conference on Learning Representations },
      year={2023},
      url={https://openreview.net/forum?id=QUaDoIdgo0}
      }

## Changelog

[2023-02-24] Pre-training code released.

[2022-06-17] Pre-trained backbone models and fine-tuned downstream detection models are now available and can be downloaded [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/rjchen_connect_hku_hk/El41ikxgHGFKr3E61JGz7JgBjITYBpO4u2WiVqYu_PxBpg?e=pItkKZ)

## Getting Started

### Installation

Please refer to [getting_started.md](docs/getting_started.md) for installation of mmdet3d. We use pytorch 1.8, mmdet 2.22.0 and mmcv 1.4.5 for this project.

### Data Preparation

* You can download DAIR-V2X dataset from [HERE](https://thudair.baai.ac.cn)
* Structure of the dataset should be as follows:
```
CO3
├── mmdet3d
├── tools
├── configs
├── data
│   ├── DAIR-V2X
│   │   ├── cooperative-dataset
│   │   │   ├── cooperative
│   │   │   ├── infrastructure-side
│   │   │   ├── vehicle-side
|   |   │   │
```
* Preprocess the dataset:
```shell
python tools/create_data.py DAIR-V2X-C
```

### Pre-training

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py configs/co3_unsupervised_representation_learning/co3.py --no-validate --launcher pytorch
```

### Downstream Evaluation

We use two main codebases for downstream evaluations and 4 3090 GPUs are used for fine-tuning. Note that to use the same backbone for evaluation, we change the original backbone in CenterPoint on Once Benchmark and you can use [this config](configs/once_centerpoint/centerpoints.yaml) to reproduce the results.
* Once Benchmark: https://github.com/PointsCoder/Once_Benchmark
* OpenPCDet: https://github.com/open-mmlab/OpenPCDet
