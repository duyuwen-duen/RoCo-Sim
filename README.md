<div align="center">

# 🚦 RoCo-Sim  

**RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation**  
🎉🎉🎉 🔥🔥🔥 RoCo-Sim was accepted to **ICCV 2025**!  

[![arXiv](https://img.shields.io/badge/arXiv-2503.10410-b31b1b)](https://arxiv.org/pdf/2503.10410)
[![HF Dataset](https://img.shields.io/badge/HuggingFace-RoCoSim__assets-orange)](https://huggingface.co/datasets/yuwendu/RoCoSim_assets)

</div>

---

## 📌 Overview

**RoCo-Sim** is a simulation framework that enriches collaborative perception datasets by overlaying 3D-rendered foreground objects onto real-world 2D backgrounds. This process improves training diversity and realism, particularly under occlusion-prone scenarios.
![image-20250313104127574](img/image-20250313104127574.png)

---

## 🧠 Pipeline Highlights

### 🗺️ 3D-to-2D Mapping & Foreground Rendering

- **Camera Extrinsic Optimization**  
  Ensures accurate 3D-to-2D projection alignment for roadside cameras.

- **Multi-View Occlusion-Aware Sampler (MOAS)**  
  Dynamically places diverse digital assets with occlusion awareness.

- **DepthSAM**  
  Models foreground-background relationships to preserve geometric consistency.

- **Scalable Post-Processing Toolkit**  
  Applies style transfer and enhancements for more realistic and enriched scenes.
![image-20250313104404046](img/image-20250313104404046.png)

---

## 🛠️ Environment Setup

**1. Create Conda Environment**

```bash
conda env create -f environment.yml
```

**2. Install Python Dependencies**

```bash
# install libcom
cd roco_sim/foreground/libcom
pip install -r requirements.txt
python setup.py install

cd libcom/controllable_composition/source/ControlCom/src/taming-transformers
python setup.py install

# install Blender
cd ../../../../../../../Blender
wget https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz
tar -xvf blender-3.5.1-linux-x64.tar.xz
rm blender-3.5.1-linux-x64.tar.xz
export blender_py=$PWD/blender-3.5.1-linux-x64/3.5/python/bin/python3.10
cd utils
# install dependency (use the -i https://pypi.tuna.tsinghua.edu.cn/simple if you are in the Chinese mainland)
$blender_py -m pip install -r requirements.txt 
$blender_py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$blender_py setup.py develop

cd ../../../..
pip install huggingface_hub==0.23.5
pip install peft==0.8.2

```

**3. Prepare Pretrained Models**

```bash
cd roco_sim/background/Depth-Anything-V2/metric_depth
mkdir checkpoints && cd checkpoints

wget -O depth_anything_v2_metric_vkitti_vitl.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true"

cd ../../../../..
cd roco_sim/background/MobileSAM/MobileSAMv2
git clone https://huggingface.co/yuwendu/MobileSAM_utils
mv MobileSAM_utils/weight .
rm -r MobileSAM_utils
cd ../../../..
```

## 🚀 Run the Demo

```
# Download Required Assets
git lfs install
git clone https://huggingface.co/datasets/yuwendu/RoCoSim_assets
mv RoCoSim_assets/data .
rm -r RoCoSim_assets


#Run the demo script and update HOME_PATH and other settings according to your environment.
bash demo.sh
```
If Blender throws an error, you can check the logs in: result/{DATA_NAME}/{SIM_DATA_NAME}/cache

--- 

## 🚀 Perform extrinsic calibration manually
(Alternatively, you may use our pre-calibrated assets available on Hugging Face to skip this process.)

The calibrated camera extrinsic parameters located at data/rcooper/calib/lidar2cam yield more accurate projection results.
(You can visualize and compare them against the official RCooper extrinsics for verification.)

If you want to perform your own calibration, you’ll need our GUI tool.
To launch it, run the following command.
A demo dataset is provided in roco_sim/calib_tools/demo_data for reference.

```
python roco_sim/calib_tools/calib_extrinsics.py
```
---

## 📂 Dataset Structure

```
standard_rcooper_mini/
└── 136-137-138-139/              # Scene containing 4 agents: 136, 137, 138, 139
    ├── 136-0/                    # Data from agent 136's camera-0
    │   ├── calib/               # Calibration files for this camera
    │   │   ├── camera_intrinsic/
    │   │   │   └── 136-0.json       # Intrinsic parameters (focal length, principal point)
    │   │   ├── lidar2cam/
    │   │   │   └── 136-0.json       # Transformation from LiDAR to camera
    │   │   └── lidar2world/
    │   │       └── 136-0.json       # Transformation from LiDAR to world/global coordinates
    │   └── train/               # Sensor data and annotations
    │       ├── image/
    │       │   └── 1693908928_315403.jpg   # RGB camera image
    │       ├── label/
    │       │   ├── camera_label/           # Annotations for 2D image (e.g., bounding boxes)
    │       │   └── lidar_label/            # Annotations for 3D point cloud (e.g., boxes in BEV)
    │       └── lidar/
    │           └── 1693908928_283546.pcd   # LiDAR point cloud (timestamped)
    ├── 137-0/
    ├── 138-0/
    ├── 139-0/
    └── coop/                   # (Optional) cooperative metadata/configurations

```

---

## 📚 Citation

If you use **RoCo-Sim** in your research, please cite:

```
@article{du2025roco,
  title={RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation},
  author={Du, Yuwen and Hu, Anning and Chao, Zichen and Lu, Yifan and Ge, Junhao and Liu, Genjia and Wu, Weitao and Wang, Lanjun and Chen, Siheng},
  journal={arXiv preprint arXiv:2503.10410},
  year={2025}
}
```

## 🙏 Acknowledgements

We would like to thank the authors and open-source contributors of the following projects, which our work builds upon or integrates:

- [**Blender**](https://www.blender.org/): Open-source 3D creation suite used for foreground rendering and scene simulation
- [**Depth-Anything-V2**](https://github.com/DepthAnything/Depth-Anything-V2): Unified monocular depth estimation framework for background depth prediction
- [**MobileSAM**](https://github.com/ChaoningZhang/MobileSAM): Lightweight segmentation model used for foreground object masking
- [**Libcom**](https://github.com/bcmi/libcom): Controllable image composition toolbox used in our simulation pipeline
- [**RCooper**](https://github.com/AIR-THU/DAIR-RCooper): Real-world large-scale dataset for roadside cooperative perception
- [**TUMTraf V2X**](https://tum-traffic-dataset.github.io/tumtraf-v2x/): Cooperative perception dataset for V2X research


If you use **RoCo-Sim** in your work, please also consider citing these foundational projects.





