# DMSPS
official code for: DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation. MedIA 2024 [MedIA](https://www.sciencedirect.com/science/article/pii/S1361841524001993?dgcid=author).
And the previous version is published on the [MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_50) 2022.

## Overall Framework
The overall framework of DMSPS：
![overall](https://github.com/HiLab-git/DMSPS/blob/master/imgs/framework.png)

## Citation
If you use this project in your research, please cite the following works:
```
@article{han2024dmsps,
  title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
  author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
  journal={Medical Image Analysis},
  pages={103274},
  year={2024},
  publisher={Elsevier}
}

@article{luo2022scribbleseg,
  title={Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision},
  author={Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang, Shaoting Zhang},
  journal={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  year={2022},
  pages={528--538}}git 
```
## Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* You can also download the specific ADCDC data used in this article from this Baidu Disk link: [ACDC](https://pan.baidu.com/s/1Wqcw_qFNezplzdewQMHXsg). The extraction code is：et38 .



## Acknowledgement
The code of scribble-supervised learning framework is borrowed from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)