
## Installation

1. Create conda environment

```shell
conda create --name defect python=3.8 -y
conda activate defect
```

2. Install `Pytorch` with cuda

```shell
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install `MMEngine` and `MMCV`

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
4. Build the source code of `MMDetection` and `DefectDetection`

```shell
cd pakcages
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
cd ../..
pip install -e . #build DefectionDetection
```

5. (Optional) Connect to your `ClearML` account 
If you don't wan't to use `ClearML`, please refer to [Visualizer](../README.md)
Install and initialize to `ClearML`

```shell
pip install clearml
clearml-init
```
Now paste your account credentials. For more information visit: https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/