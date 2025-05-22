# Environment Setup

```bash
conda create -n vlsat python=3.8
conda activate vlsat
pip install -r requirement.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-geometric==2.2.0
pip install git+https://github.com/openai/CLIP.git
pip install hydra
pip install hydra-core --upgrade --pre
```

---

# Dataset Preparation

## 1. Download 3RScan  
First, download the 3RScan dataset. You can follow the instructions provided in the [3DSSG official guide](https://github.com/ShunChengWu/3DSSG#preparation).

## 2. Generate 2D Multi-view Images  
Convert the point clouds into 2D images from multiple viewpoints. Make sure to update the internal path in the script to match your local environment.

```bash
# Modify the path in pointcloud2image.py to match your local environment.
python data/pointcloud2image.py
```

## 3. Directory Structure  
Make sure your folders are organized as follows for proper operation:

```
data
  3DSSG_subset
    relations.txt
    classes.txt

  3RScan
    <scan_id_1>
      multi_view/
      labels.instances.align.annotated.v2.ply
    <scan_id_2>
    ...
```

---

# Training & Evaluation

```bash
# Train
python -m main --mode train --config <config_path> --exp <exp_name>

# Evaluate
python -m main --mode eval --config <config_path> --exp <exp_name>
```

---

# References

This project is inspired by and partially based on the following repositories:

- [3DSSG](https://github.com/ShunChengWu/3DSSG)
- [VL-SAT (CVPR 2023)](https://github.com/wz7in/CVPR2023-VLSAT)