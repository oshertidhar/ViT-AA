# ViT-AA
TAU DeepLearning course project. Created By Osher Tidhar and Yoav Kurtz

---
This repository includes our additions to the [DEiT training code](https://github.com/facebookresearch/deit).

## General
- Our ViT+AA model is implemented in `AAVitFactory.py`.
- Needed extensions for training the model on CIFAR10 dataset are added to `main.py, datasets.py` which were taken from DEiT repository.
- `AAVitFactory.py` creates models according to configuration files found in `cfgs` folder.

## How to run

1. Follow instructions in DeIT [git repo](https://github.com/facebookresearch/deit).
2. Place `AAVitFactory.py` in the cloned DeIT repo.
3. Replace `main.py, datasets.py` with the corresponding files in this repository.
4. ViT-AA model can be trained by executing `main.py`.

## Citations
```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
```bibtex
@misc{touvron2020training,
    title   = {Training data-efficient image transformers & distillation through attention}, 
    author  = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
    year    = {2020},
    eprint  = {2012.12877},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
