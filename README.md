# DL project (CS4240): Reproducibility of KPRNet
This repository contains the source code for a reproduction/extension of the paper: KPRNet: Improving projection-based LiDAR
semantic segmentation.

**Autors:** \
*Group 17:* \
Aden Westmaas (4825373) \
Badr Essabri (5099412) \
Guido Dumont (5655366)

**Documentation:** \
Blog post: [Project blog post](https://hackmd.io/llz5iQ6bSl-jkUuuQce0ng?both) \
Original paper: [KPRNet: Improving projection-based LiDAR semantic segmentation](https://arxiv.org/pdf/2007.12668.pdf) \
Original GitHub: [KPRNet github repository](https://github.com/DeyvidKochanov-TomTom/kprnet)

## Installation
Please follow the installation instruction of the original [repository](https://github.com/DeyvidKochanov-TomTom/kprnet) and add the packages listed in *requirements.txt* afterwards. 

## Repository content
1. Reproduction of the results in the paper
2. Data augmentation to check robustness
3. Implementation of the KITTI-360 dataset

### Reproduction of the results in the paper
Run *run_inference.py* in the source code in the original [repository](https://github.com/DeyvidKochanov-TomTom/kprnet).

### Data augmentation to check robustness
In the file semantic_kitti.py, the percentage of pixel dropout can be changed in [this line](https://github.com/guidodumont/DL-project/blob/37a7876d4b64270cc9bdb75a2d67d976ea9446ba/kprnet/datasets/semantic_kitti.py#L128). \
Run *run_inference.py* in the source code in the "Badr" branch of this repository.

### Reproduction of the results in the paper
Run the Jupyter Notebook *implementation_kitti360.ipynb*.
