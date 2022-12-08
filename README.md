# DenseTeacher

This project provides an implementation for our ECCV2022 paper "[DenseTeacher: Dense Pseudo-Label for Semi-supervised Object Detection](https://arxiv.org/abs/2207.02541v2)" on PyTorch.

<img src="./illustration.png" width="700" height="330">


## Requirements
* [cvpods](https://github.com/Megvii-BaseDetection/cvpods)

## Get Started

* install cvpods locally (requires cuda to compile)
```shell

python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

# Or,
pip install -r requirements.txt
python3 setup.py build develop
```

* prepare datasets
```shell
cd /path/to/cvpods/datasets
ln -s /path/to/your/coco/dataset coco

```
* start training
```shell
cd DenseTeacher/coco-p10
pods_train --dir .
# Evaluation will be automatically start after each epoch
```



## Acknowledgement
This repo is developed based on cvpods. Please check [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.
