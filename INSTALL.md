## Installation

### Requirements

* Linux or macOS with Python ≥ 3.8
* PyTorch ≥ 1.9
* torchvision compatible with the installed PyTorch version
* Detectron2
* OpenCV (optional, for visualization)
* CUDA Toolkit (for MSDeformAttn compilation)

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### Install Detectron2

Please follow the official Detectron2 installation instructions:

https://detectron2.readthedocs.io/tutorials/install.html

---

### Compile MSDeformAttn

After installing the dependencies, compile the CUDA operators:

```bash
cd m2fp/modeling/pixel_decoder/ops
sh make.sh
```

`CUDA_HOME` must point to your CUDA installation directory.

---

### Example Conda Environment

```bash
conda create --name qhisnet python=3.8 -y
conda activate qhisnet

conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install -U opencv-python

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..

git clone https://gitee.com/yufeixu0507/pedestrian-segmentation.git

cd pedestrian-segmentation

pip install -r requirements.txt

cd m2fp/modeling/pixel_decoder/ops
sh make.sh
```

---

### Verification

```bash
python train_net.py --help
```

If the command executes successfully, the installation is complete.
