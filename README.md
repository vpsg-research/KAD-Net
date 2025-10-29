# KAD-Net: Kolmogorov-Arnold and Differential-Aware Networks for Robust and Sensitive Proactive Deepfake Forensics

This repository provides the implementation of the paper "KAD-Net: Kolmogorov-Arnold and Differential-Aware Networks for Robust and Sensitive Proactive Deepfake Forensics".
We are grateful for any feedback and contributions. It must be acknowledged that this work still has several limitations. We hope this codebase is useful to your research. Best wishes!

---


## 📑 Contents

- [TODO](#-todo)
- [Environment Setup](#-environment-setup)
- [Datasets](#-datasets)
- [Noise Pool](#-noise-pool)
- [Training](#-training)
- [Testing](#-testing)
- [Visual Results](#-visual-results)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)
- [License](#-license)



---

## ☑️ TODO
- [√] Project page released
- [√] Dataset preparation instructions released
- [√] Release of core implementation
- [√] Release of training and evaluation scripts
- [√] Pretrained model and demo

---



## 🖥️ Environment Setup

```bash
conda env create -f requirements.yml
```


---


## 📁 Datasets
KAD-Net was trained on the CelebA-HQ dataset and evaluated on CelebA-HQ, CelebA, and LFW datasets at resolutions of 128×128 and 256×256. We do not own these datasets; they can be obtained from their respective official websites.  
  [🔗 CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)  
  [🔗 CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
  [🔗 LFW](http://vis-www.cs.umass.edu/lfw/)  

---

## 🌪️ Noise Pool
The noise pool includes two types of distortions:

1. **Common image distortions**: `Identity`, `JPegTest`, `Resize`, `GaussianBlur`, `MedianBlur`, `Brightness`, `Contrast`, `Saturation`, `Hue`, `Dropout`, `SaltPepper`, `GaussianNoise`.

2. **Deepfake-related malicious distortions**:   
[🔗 SimSwap](https://arxiv.org/abs/2106.06340): Identity-aware face swapping with high-quality alignment.  
[🔗 GANimation](https://arxiv.org/abs/1807.09251): Expression manipulation via conditional adversarial networks.   
[🔗 StarGAN](https://arxiv.org/abs/1801.00699): Multi-domain attribute translation (e.g., hair color, age, gender).  

---


## 🏋️‍♂️ Training
**Configuration (required).**  
Edit `KAD-Net/cfg/train_KAD_Net.yaml` **before** running any command.  
This YAML controls dataset paths, image size, watermark length, model hyper-parameters, optimizer/schedule, noise layers, and checkpoint. Ensure that the YAML settings are consistent with your experimental setup.

```bash
python train.py
```
---

## 🔎 Testing
**Configuration (required).**  
Before testing, edit `KAD-Net/cfg/test_KAD_Net.yaml`. This YAML specifies the test dataset paths, image size, message length, noise layers options and the checkpoint(s) to load.   
Make sure `image_size` and `message_length` are consistent with training.

**Checkpoint layout (by convention):**

```text
KAD-Net/results/
├─ ST/
│  ├─ 128/
│  │   └─ ST_KAD_Net/
│  │       ├─ D_XX.pth   # e.g., Discriminator (use your project's actual naming)
│  │       └─ EC_XX.pth  # e.g., Encoder and Decoder (XX denotes the epoch number)
│  └─ 256/
│      └─ ST_KAD_Net/
│          ├─ D_XX.pth
│          └─ EC_XX.pth
└─ FD/
   ├─ 128/
   │   └─ FD_KAD_Net/
   │       ├─ D_XX.pth
   │       └─ EC_XX.pth
   └─ 256/
       └─ FD_KAD_Net/
           ├─ D_XX.pth
           └─ EC_XX.pth
```

**Where to place/load checkpoints.**  
We keep model checkpoints under the following directories:
- ST branch (SourceTracer): `KAD-Net/results/ST/128` or `KAD-Net/results/ST/256`  
- FD branch (ForgeryDetector): `KAD-Net/results/FD/128` or `KAD-Net/results/FD/256`  
Pre-trained model checkpoints are available in the aforementioned directories to enable direct testing and use. If you would like to use our pre-trained models directly, please click [here](https://drive.google.com/drive/folders/1p2no9qLXh4rwcmViAZEWKPWJSSPPkzoJ?usp=sharing) to download and run them.

```bash
python test.py
```

---

## 🔍 Visual Results
<p align="center">
  <img width="800" src="pictures\fig6.png">
</p>
<p align="center"><em>Visualization of various noise effects applied to the image, including common distortions and deepfake manipulations. The first row is the original image, the second row is the watermarked image, and the third row is the watermarked image after noise distortion. The fourth row is the extracted watermark residual signal, and the fifth row is the impact of noise distortion on the watermark residual signal.</em></p>

---

## 🧾 Acknowledgements
This work is inspired by remarkable studies such as
[Kolmogorov–Arnold Networks (KAN)](https://openreview.net/forum?id=Ozo7qJ5vZi),
[EditGuard](https://arxiv.org/pdf/2312.08883),
[SepMark](<https://dl.acm.org/doi/10.1145/3581783.3612471>),
and
[CDCN](<https://ieeexplore.ieee.org/document/9156660>).
We extend our sincere thanks to all the contributors of these works.

---

## 📚 Citation

If you find this repository helpful, please cite our paper:

```bibtex
@article{HE2025114692,
  title   = {KAD-Net: Kolmogorov-Arnold and Differential-Aware Networks for Robust and Sensitive Proactive Deepfake Forensics},
  author  = {Sijia He and Yunfeng Diao and Yongming Li and Chen Sun and Liejun Wang and Zhiqing Guo},
  journal = {Knowledge-Based Systems},
  year    = {2025},
}
```
---

## 🧾 License

This repository is released under the [Apache 2.0 License](LICENSE).

---
