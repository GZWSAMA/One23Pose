Here is the **fully English version** of your **README**, cleaned up, well-structured, and ready for use in a professional or open-source project.

---

# 🎯 One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation

> **One-2-3-Pose**: A Fast and Accurate Pipeline for 6D Pose and Scale Estimation from a Single Image

---

![Teaser](assets/images/teaser.png)

---

## 📄 Abstract

We introduce **One-2-3-Pose**, a fast and accurate pipeline for 6D object pose and scale estimation from a single image. Given just one image of an object, our system can:

- Reconstruct its 3D model from scratch
- Estimate its 3D position, orientation, and scale with high precision

This approach leverages 3D generation and domain randomization to enable **one-shot 6D pose estimation** without requiring any 3D annotations or prior object models.

---

## ⚙️ Installation

> **Note**: This project is developed and tested under **Python 3.11**.

### 1. Install PyTorch with CUDA support

Install PyTorch with the version compatible with your CUDA toolkit. Example for CUDA 12.1:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install Boost and Eigen Libraries

```bash
conda install -c conda-forge boost
conda install -c conda-forge eigen=3.4.0
```

---

### 4. Clone and Install Extensions

```bash
mkdir -p tmp/extensions

# DiffOctreeRaster
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git tmp/extensions/diffoctreerast
pip install tmp/extensions/diffoctreerast

# Mip-Splatting
git clone https://github.com/autonomousvision/mip-splatting.git tmp/extensions/mip-splatting
pip install tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# PyTorch3D
pip install git+https://github.com/facebookresearch/pytorch3d.git
```

---

### 5. Build F-Pose

```bash
cd one23pose/fpose/fpose
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
cd ../../..
```

---

### 6. Install Packages in Development Mode

```bash
# Install F-Pose
cd one23pose/fpose
pip install -e .
cd ..

# Install SAM2-in-Video
cd ../SAM2-in-video
pip install -e .
cd ..

# Install Trellis
cd ../trellis
pip install -e .
cd ..

# Install SpaTrackerV2
cd ../SpaTrackerV2
pip install -e .
cd ..
```

---

### 7. Download Pretrained Weights

```bash
bash one23pose/scripts/download_weights.sh
```

---

### 8. Patch Transformers Library

Edit the following file:

```
$CONDA_PREFIX/lib/python3.11/site-packages/transformers/models/sam/processing_sam.py
```

**Change line 121** to:

```python
original_sizes = original_sizes.cpu().numpy()
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
TODO
```

---

## 📝 Notes

- This project depends on multiple third-party libraries and modules. It is highly recommended to use a virtual environment (e.g., `conda`) for development.
- If you encounter any installation issues, refer to the official documentation of each sub-module or open an issue.
- This README will be continuously updated and improved as the project evolves.
