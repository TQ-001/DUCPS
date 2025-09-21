# DUCPS
Deep Unfolded Cauchy Proximal Splitting


## 📁 Data Structure

Your **preprocessed** (subsampled) training/validation data should be organized as:

```

Data/
├── random1/
│   ├── Fold1/
│   │   ├── train/
│   │   └── val/
│   ├── Fold2/
│   │   ├── train/
│   │   └── val/
│   └── Fold3/
│       ├── train/
│       └── val/
└── random2/
├── Fold1/
│   ├── train/
│   └── val/
├── Fold2/
│   ├── train/
│   └── val/
└── Fold3/
├── train/
└── val/

````

- `random1`, `random2` are different random splits.
- Each `Fold{1,2,3}` contains `train/` and `val/` subsets.


---

## 🧪 Data Preparation (Subsampling)

Training and validation data are **preprocessed offline via subsampling** to avoid doing it on-the-fly during training.

Use `subsampler.py` to create the subsampled datasets and save them into the structure above.

---
## ⚙️ Training Parameters

Key arguments:

| Argument / Flag       | Description                                         | Notes                                            |
| --------------------- | --------------------------------------------------- | ------------------------------------------------ |
| `--tv`                | Weight for TV (Total Variation) loss                | Appears in results path as `TV{ARGS.tv}`         |
| `--cauchy`            | Weight for Cauchy loss                              | Appears in results path as `Cauchy{ARGS.Cauchy}` |
| `--iter`              | Number of iterations (e.g., unrolling / steps)      | Controls optimization depth                      |
| `--disable_attention  | Choose to use attention or not                      | May **increase computation time**                |
| `--skip`              | Enable skip connections                             |                                                  |

> **Save-path consistency:** The results directory uses
> `TV{ARGS.tv}_Cauchy{ARGS.Cauchy}_att{ARGS.disable_attention}_skip{ARGS.skip}`.
---

## ▶️ Running Training

If your entry point is `autorun.py`:

```bash
python autorun.py 
```

## 💾 Outputs

Trained models are saved under:

```
./trained_model/TV{ARGS.tv}_Cauchy{ARGS.Cauchy}_att{ARGS.disable_attention}_skip{ARGS.skip}/
```

*(The exact folder name depends on your argument values and how your script formats `ARGS`.)*

---

## 📓 Logging with Weights & Biases

Training logs are recorded with **Weights & Biases**.
You’ll need a W\&B account and to log in locally.

* W\&B docs: [https://docs.wandb.ai/guides/](https://docs.wandb.ai/guides/)

**Quick start:**

```bash
pip install wandb
wandb login
```

Log **locally first** and upload later using `upload_logs.py`:

```bash
python upload_logs.py
```

---

## 🛠️ Environment Setup

1. **Create & activate an environment**

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

2. **Install PyTorch** (match your hardware)

```bash
# Example: CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Example: CUDA 12.1 (adjust for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Run the model**

```bash
python autorun.py
```
---

## ✅ Notes & Tips

* Using attention (`--att true`) may **increase runtime and memory**.
* Generate subsampled datasets **before** training to avoid online preprocessing.

---

## 📄 Key References

[1] Karakus O, Anantrasirichai N, Aguersif A, Silva S, Basarab A, Achim A. Detection of Line Artifacts in Lung Ultrasound Images of COVID-19 Patients Via Nonconvex Regularization. IEEE Trans Ultrason Ferroelectr Freq Control. 2020 Nov;67(11):2218-2229. doi: 10.1109/TUFFC.2020.3016092. Epub 2020 Aug 12. PMID: 32784133; PMCID: PMC8544933.

[2] Huang T, Li S, Jia X, et al. Neighbor2neighbor: Self-supervised denoising from single noisy images[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 14781-14790.

---

