# Installation

This code is based on [Detectron2](https://github.com/facebookresearch/detectron2).

Use the following instructions.

#### Create a new conda environment:

```conda create -n bodyhands python=3.7```

```conda activate bodyhands```

#### Install the following dependencies:

```conda install pytorch torchvision cudatoolkit=10.0 -c pytorch```

```python -m pip install python-dateutil>=2.1 pycocotools>=2.0.1```

```python -m pip install opencv-python ipython scipy scikit-image```

#### Clone Detectron2 v0.1.1 and install the following commit (using other commit can give errors):

```git clone https://github.com/facebookresearch/detectron2.git --branch v0.1.1 bodyhands```

```cd bodyhands```

```git checkout db1614e```

```python -m pip install -e .```

#### Clone this BodyHands repository:

```git clone git@github.com:SupreethN/BodyHands_Private.git```

```cd BodyHands_Private```
