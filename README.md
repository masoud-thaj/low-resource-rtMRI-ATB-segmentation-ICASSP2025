# Low-resource-rtMRI-ATB-segmentation-ICASSP2025

## 🔊 Role of the Pretraining and the Adaptation Data Sizes for Low-Resource rtMRI Video Segmentation

Official implementation of the paper:  
**"Role of the Pretraining and the Adaptation Data Sizes for Low-Resource rtMRI Video Segmentation"**  
Published at **ICASSP 2025**

**Authors:** Masoud Thajudeen Tholan, Vinayaka Hegde, Chetan Sharma, Prasanta Kumar Ghosh  
📄 [Link to Paper (IEEE)](https://doi.org/10.1109/ICASSP49660.2025.10889096)

---

## 📌 Overview

This repository contains the official codebase for our ICASSP 2025 paper:  
*"Role of the Pretraining and the Adaptation Data Sizes for Low-Resource rtMRI Video Segmentation"*.

Real-time Magnetic Resonance Imaging (rtMRI) is widely used in speech production studies due to its ability to provide a dynamic view of the vocal tract during articulation. In this work, we investigate the role of pretraining and adaptation data sizes in low-resource settings using the SegNet model for Air-Tissue Boundary (ATB) segmentation.

Our findings demonstrate:
- The effectiveness of model fine-tuning with limited rtMRI data.
- Adaptation is possible with as few as **15 labeled frames** from new speakers.
- Pretraining and adaptation strategies significantly influence segmentation quality.

---

## 📁 Repository Structure & Code Usage

The repository contains six major components corresponding to the experiments described in the paper:

1. **`pretrain.py/`**  
   Train models using 8 out of 10 subjects from the **USC-TIMIT** dataset.

2. **`tune.py/`**  
   Fine-tune pretrained models using a small number of frames from the remaining 2 subjects (**F5 and M5**).

3. **`prediction.py/`**  
   Run inference using the pretrained and fine-tuned models on 3 unseen videos of F5 and M5.

4. **`tune_for_74.py/`**  
   Fine-tune pretrained models using a small subset of data from 2 subjects from the **USC 75-SPEAKER** dataset (we refer them as **F6 and M6**).

5. **`prediction_for_74.ipynb/`**  
   Run inference on unseen videos of F6 and M6 using the models trained in step 4.

6. **`pretrain_for_74.py/`**  
   Train a "matched condition" model using more labeled data from F6 and M6 to assess the benefit of fine-tuning.

📊 All results are visualized and plotted as shown in the paper.

---

## 📦 Main Branch

All source code for the above six components is located in the `main` branch of this repository.

---

## 🔖 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10889096,
  author={Tholan, Masoud Thajudeen and Hegde, Vinayaka and Sharma, Chetan and Ghosh, Prasanta Kumar},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Role of the Pretraining and the Adaptation data sizes for low-resource real-time MRI video segmentation}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889096}
}
```

## 📄 License

This code is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


