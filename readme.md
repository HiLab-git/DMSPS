# DMSPS
official code for: DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation. MedIA 2024 [MedIA](https://www.sciencedirect.com/science/article/pii/S1361841524001993?dgcid=author).
And the previous vervision is published on the [MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_50) 2022.

### Overall Framework
There are three branches based on different attention mechanisms and two losses in our framework
![overall](https://github.com/HiLab-git/DMSPS/blob/main/imgs/framework.png)




### Citation
```
@article{han2024dmsps,
  title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
  author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
  journal={Medical Image Analysis},
  pages={103274},
  year={2024},
  publisher={Elsevier}
}

@article{zhong2024semi,
  title={Semi-supervised pathological image segmentation via cross distillation of multiple attentions and Seg-CAM consistency},
  author={Zhong, Lanfeng and Luo, Xiangde and Liao, Xin and Zhang, Shaoting and Wang, Guotai},
  journal={Pattern Recognition},
  pages={110492},
  year={2024},
  publisher={Elsevier}
}
```

### Acknowledgement
The code of scribble-supervised learning framework is borrowed from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)