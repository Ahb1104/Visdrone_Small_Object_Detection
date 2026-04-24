# VisDrone Small Object Detection

Experiments comparing three architectures for small object detection on the [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) dataset.

**Repository:** https://github.com/Ahb1104/Visdrone_Small_Object_Detection

---

## Repository Structure

```
Visdrone_Small_Object_Detection/
│
├── EDA/
│   └── eda-visdrone-dataset.ipynb          # Dataset exploration notebook
│
├── Models/
│   ├── YOLO_Small_DNNHead.ipynb            # YOLOv8s-p2 + TinyVisDroneHead (3 variations)
│   ├── ResNet50_frcnn_DNN.py               # ResNet-50 + Faster R-CNN + Custom DNN head
│   └── SwinT_E2E_VisDrone.ipynb            # Swin-T + Faster R-CNN + TransformerBoxHead
│
├── Scripts/
│   ├── run_frcnn_end_to_end.sh             # ResNet end-to-end training launcher
│   ├── run_frcnn_backbone_only.sh          # ResNet pretrained backbone baseline launcher
│   └── run_frcnn_frozen_backbone.sh        # ResNet frozen-backbone training launcher
│
├── env.yml                                 # Conda environment spec (OOD cluster)
└── setup.sh                                # One-shot environment setup script (OOD cluster)
```

| Model | Run environment | Entry point |
|---|---|---|
| YOLOv8s-p2 + TinyVisDroneHead | Google Colab (Kaggle data) | `Models/YOLO_Small_DNNHead.ipynb` |
| Swin-T + Faster R-CNN + TransformerBoxHead | Google Colab | `Models/SwinT_E2E_VisDrone.ipynb` |
| ResNet-50 + Faster R-CNN + Custom DNN head | Northeastern OOD Explorer cluster | `Models/ResNet50_frcnn_DNN.py` via `Scripts/*.sh` |

---

## Dataset Sources

All models use **VisDrone2019-DET**, sourced from the official VisDrone GitHub organization:

| Split | Download |
|---|---|
| Train (~1.4 GB) | https://github.com/VisDrone/VisDrone-Dataset |
| Validation (~0.1 GB) | https://github.com/VisDrone/VisDrone-Dataset |
| Test-dev (~0.5 GB) | https://github.com/VisDrone/VisDrone-Dataset |

The dataset contains 10,209 static aerial images annotated with 10 object categories across 14 cities.

**Where data lives by platform:**

- **YOLO Colab notebook** — Kaggle API downloads and unzips the dataset automatically into the Colab runtime at `/content/datasets/VisDrone/`. No manual download needed.
- **Swin-T Colab notebook** — dataset is downloaded directly inside the notebook from the VisDrone GitHub release URLs into `/content/VisDrone/`. No manual download needed.
- **OOD cluster (ResNet)** — data must be manually downloaded to `/scratch/<your_username>/VisDrone_YOLO/data/VisDrone/` before submitting any jobs. See [Step 2 of the OOD section](#2-download-the-visdrone-dataset).

---

## Running on Google Colab

### A. YOLO — `Models/YOLO_Small_DNNHead.ipynb`

Trains three variations of YOLOv8s-p2 augmented with a lightweight `TinyVisDroneHead`: baseline, end-to-end fine-tuning, and frozen-backbone. Data is pulled from Kaggle.

#### Prerequisites

- A **Kaggle account** (free)
- **Google Colab** access (the notebook was built and tested on Colab)
- **GPU runtime** — an **H100 or A100 80 GB** (Colab Pro+) is strongly recommended. The pipeline requires **>50 GB GPU VRAM** to run all three variations at full batch size. An A100 40 GB or L4 (24 GB) is not sufficient at default settings.

#### 1. Get your Kaggle API token

1. Go to [kaggle.com](https://kaggle.com) and log in.
2. Click your profile picture (top right) → **Settings**.
3. Scroll to the **API** section and click **Create New Token**.
4. A popup shows a token string that starts with `KGAT_` (looks like `KGAT_82514c3ab7bf...`).
5. **Copy this token** — you won't be able to view it again, but you can always generate a new one.

> ⚠️ Treat this token like a password. Don't paste it into chats, commits, or shared notebooks.

#### 2. Open the notebook in Colab

**Option A — open directly from GitHub:**

1. In the GitHub file view, click `Models/YOLO_Small_DNNHead.ipynb`.
2. Click the **"Open in Colab"** button (or prepend `https://colab.research.google.com/github/` to the repo URL).

**Option B — upload manually:**

1. Download the `.ipynb` file from GitHub.
2. Go to [colab.research.google.com](https://colab.research.google.com) → **File** → **Upload notebook**.

#### 3. Enable a high-memory GPU

**Runtime** → **Change runtime type** → set **Hardware accelerator** to **GPU** → expand **GPU type** and select:

- **H100** (80 GB VRAM) — preferred; requires Colab Pro+
- **A100** (80 GB VRAM) — also works; requires Colab Pro+

> ⚠️ Always confirm you have >50 GB available by running `!nvidia-smi` in the first cell before proceeding.

Click **Save**.

#### 4. Run the notebook

Run cells in order. When you reach the dataset-download cell, it will prompt:

```
Paste your KAGGLE_API_TOKEN (starts with KGAT_):
```

Paste the token you copied and press Enter. The input is masked so it won't appear in notebook output — safe to share the notebook afterward.

The dataset (~2 GB) downloads and unzips automatically. Every subsequent cell runs end-to-end without further input.

#### Expected runtime

| GPU | Full 30-epoch run (per variation) |
|---|---|
| T4 (free, 16 GB) | ~4 hours — OOM likely at default batch size |
| A100 / H100 80 GB (Pro+) | ~1 hour |

The notebook trains three variations, so total time is 3× the per-variation time.

#### Troubleshooting

- **"Kaggle API token invalid"** — the token may have been revoked or contain extra whitespace. Generate a new one from kaggle.com → Settings → API and try again.
- **Colab disconnects mid-training** — free-tier sessions time out after ~12 hours of idle use. Mount Google Drive at the top of the notebook and point `project=` to a Drive path to save checkpoints.
- **OOM errors** — reduce `batch=24` to `batch=12` or `batch=8` in the training cell. On a T4, go as low as `batch=4`.

---

### B. Swin-T — `Models/SwinT_E2E_VisDrone.ipynb`

Trains a Swin Transformer backbone paired with Faster R-CNN and a custom `TransformerBoxHead` in an end-to-end configuration. Data is downloaded directly inside the notebook from VisDrone's GitHub releases — no Kaggle account required.

#### Prerequisites

- **Google Colab** access
- **GPU runtime** — select a GPU with **>50 GB VRAM** (H100 or A100 80 GB, Colab Pro+). The Swin-T backbone is more memory-intensive than YOLOv8s at equivalent batch sizes.

#### 1. Open the notebook in Colab

**Option A — open directly from GitHub:**

1. In the GitHub file view, click `Models/SwinT_E2E_VisDrone.ipynb`.
2. Click the **"Open in Colab"** button (or prepend `https://colab.research.google.com/github/` to the repo URL).

**Option B — upload manually:**

1. Download the `.ipynb` file from GitHub.
2. Go to [colab.research.google.com](https://colab.research.google.com) → **File** → **Upload notebook**.

#### 2. Enable a high-memory GPU

**Runtime** → **Change runtime type** → **Hardware accelerator: GPU** → select:

- **H100** (80 GB VRAM) — preferred; requires Colab Pro+
- **A100** (80 GB VRAM) — also works; requires Colab Pro+

> ⚠️ Confirm >50 GB is available with `!nvidia-smi` before running training cells.

#### 3. Run the notebook

Run cells in order. The notebook downloads the VisDrone2019-DET train, val, and test-dev splits directly from the VisDrone GitHub release URLs into `/content/VisDrone/` — no token or manual download step is required. From there, all subsequent cells run end-to-end.

#### Expected runtime

| GPU | Approximate full training run |
|---|---|
| A100 / H100 80 GB (Pro+) | ~2–3 hours |

#### Troubleshooting

- **Colab disconnects mid-training** — mount Google Drive and save checkpoints periodically. Add a checkpoint callback pointing to a Drive path near the top of the notebook.
- **OOM errors** — reduce the batch size in the training configuration cell. The Swin-T backbone is sensitive to image resolution; reducing `img_size` from 1024 to 640 can significantly cut memory usage.
- **Slow dataset download** — if the VisDrone GitHub download stalls, re-run that cell. Alternatively, pre-download the splits to Drive and update the data path in the notebook.

---

## Running on Northeastern OOD Explorer (ResNet-50)

`Models/ResNet50_frcnn_DNN.py` implements ResNet-50 + Faster R-CNN with a custom DNN detection head. The three shell scripts in `Scripts/` each launch a different training variation as a SLURM batch job on the **Northeastern Discovery / OOD Explorer cluster**.

| Script | What it runs |
|---|---|
| `run_frcnn_end_to_end.sh` | Full end-to-end training — all layers unfrozen |
| `run_frcnn_backbone_only.sh` | Pretrained backbone baseline — standard Faster R-CNN, no custom head |
| `run_frcnn_frozen_backbone.sh` | Frozen ResNet-50 backbone — only the DNN head trains |

### Prerequisites

- An active **Northeastern account** with cluster access
- Access to [ood.discovery.neu.edu](https://ood.discovery.neu.edu) (Open OnDemand portal)
- Your **NU username** — shown as `<your_username>` throughout (e.g., `doe.j`)

---

### 1. Create your directory trees

`setup.sh` expects `env.yml` at `~/visdrone/env.yml` (home directory). The dataset and all training outputs live in scratch. Log in to the cluster via OOD → **Clusters** → **Discovery Shell Access**, then run:

```bash
# Home directory — for env files (persists across sessions, small footprint)
mkdir -p ~/visdrone

# Scratch — for data, outputs, and logs (large files, not backed up)
mkdir -p /scratch/<your_username>/VisDrone_YOLO/data/VisDrone
mkdir -p /scratch/<your_username>/VisDrone_YOLO/outputs/frcnn_end_to_end
mkdir -p /scratch/<your_username>/VisDrone_YOLO/outputs/frcnn_backbone_only
mkdir -p /scratch/<your_username>/VisDrone_YOLO/outputs/frcnn_frozen_backbone
mkdir -p /scratch/<your_username>/VisDrone_YOLO/logs
mkdir -p /scratch/<your_username>/VisDrone_YOLO/weights
```

Clone the repository into scratch and copy the env files to your home directory:

```bash
cd /scratch/<your_username>/VisDrone_YOLO
git clone https://github.com/Ahb1104/Visdrone_Small_Object_Detection.git .

# setup.sh reads env.yml from ~/visdrone/ — copy both there
cp env.yml ~/visdrone/env.yml
cp setup.sh ~/visdrone/setup.sh
```

Your final layout across both locations:

```
~/visdrone/                                 ← home dir (persists, lightweight)
├── env.yml                                 ← conda environment spec
└── setup.sh                                ← one-shot environment setup script

/scratch/<your_username>/VisDrone_YOLO/     ← scratch (large files, purged periodically)
├── Models/
│   └── ResNet50_frcnn_DNN.py
├── Scripts/
│   ├── run_frcnn_end_to_end.sh
│   ├── run_frcnn_backbone_only.sh
│   └── run_frcnn_frozen_backbone.sh
├── data/
│   └── VisDrone/
│       ├── VisDrone2019-DET-train/
│       │   ├── images/
│       │   └── annotations/
│       ├── VisDrone2019-DET-val/
│       │   ├── images/
│       │   └── annotations/
│       └── VisDrone2019-DET-test-dev/
│           ├── images/
│           └── annotations/
├── outputs/
│   ├── frcnn_end_to_end/       ← checkpoints + metrics from end-to-end run
│   ├── frcnn_backbone_only/    ← checkpoints + metrics from backbone-only run
│   └── frcnn_frozen_backbone/  ← checkpoints + metrics from frozen-backbone run
├── logs/                       ← SLURM stdout/stderr (.out and .err files)
└── weights/                    ← pretrained backbone weights (if applicable)
```

---

### 2. Download the VisDrone dataset

```bash
cd /scratch/<your_username>/VisDrone_YOLO/data/VisDrone

# Train split (~1.4 GB)
wget -O VisDrone2019-DET-train.zip \
  "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip"

# Validation split (~0.1 GB)
wget -O VisDrone2019-DET-val.zip \
  "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip"

# Test-dev split (~0.5 GB)
wget -O VisDrone2019-DET-test-dev.zip \
  "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"

unzip VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-val.zip
unzip VisDrone2019-DET-test-dev.zip
```

---

### 3. Insert your username into the `.sh` scripts

Each `.sh` file in `Scripts/` contains `<your_username>` placeholders in its SLURM directives and path variables. You must replace them before submitting. Repeat for all three scripts:

```bash
nano /scratch/<your_username>/VisDrone_YOLO/Scripts/run_frcnn_end_to_end.sh
nano /scratch/<your_username>/VisDrone_YOLO/Scripts/run_frcnn_backbone_only.sh
nano /scratch/<your_username>/VisDrone_YOLO/Scripts/run_frcnn_frozen_backbone.sh
```

In each file, update every line that contains `<your_username>`:

```bash
# BEFORE
#SBATCH --output=/scratch/<your_username>/VisDrone_YOLO/logs/%j.out
#SBATCH --error=/scratch/<your_username>/VisDrone_YOLO/logs/%j.err
DATA_ROOT=/scratch/<your_username>/VisDrone_YOLO/data/VisDrone
OUTPUT_DIR=/scratch/<your_username>/VisDrone_YOLO/outputs/frcnn_end_to_end

# AFTER  (example for username doe.j)
#SBATCH --output=/scratch/doe.j/VisDrone_YOLO/logs/%j.out
#SBATCH --error=/scratch/doe.j/VisDrone_YOLO/logs/%j.err
DATA_ROOT=/scratch/doe.j/VisDrone_YOLO/data/VisDrone
OUTPUT_DIR=/scratch/doe.j/VisDrone_YOLO/outputs/frcnn_end_to_end
```

Save and close (`Ctrl+O` → Enter → `Ctrl+X` in nano). Repeat for the other two scripts, updating `OUTPUT_DIR` to match each script's variation name.

---

### 4. Load required modules

`setup.sh` loads these exact module versions — make sure they match if you ever run commands manually:

```bash
module load cuda/12.1.1
module load anaconda3/2024.06
```

To avoid repeating this every session, add these two lines to your `~/.bashrc`.

---

### 5. Set up the Python environment (first time only)

The repository includes `setup.sh` and `env.yml` which handle the entire environment setup in one shot. Run `setup.sh` directly from the shell — it is **not** a SLURM script and should not be submitted with `sbatch`.

```bash
bash ~/visdrone/setup.sh
```

This script does the following automatically:

1. Loads `cuda/12.1.1` and `anaconda3/2024.06`
2. Creates (or updates) a conda environment named `visdrone` from `env.yml`
   - Python 3.10–3.12, numpy, matplotlib, tqdm, jupyter, pillow
3. Installs `torch==2.6.0` and `torchvision==0.21.0` for CUDA 12.1
4. Installs `timm`, `torchmetrics`, `ultralytics`, and `ipykernel`
5. Registers a Jupyter kernel as **"Python (visdrone)"** so it's available in OOD Jupyter sessions
6. Prints a CUDA sanity check confirming GPU is visible

When it finishes you should see output like:

```
PyTorch : 2.6.0+cu121
CUDA    : True
GPU     : NVIDIA A100-SXM4-80GB
=== Done — activate with: source activate visdrone ===
```

If you need to activate the environment manually in a later session:

```bash
module load cuda/12.1.1 anaconda3/2024.06
source activate visdrone
```

---

### 6. Make the scripts executable

```bash
chmod +x /scratch/<your_username>/VisDrone_YOLO/Scripts/*.sh
```

---

### 7. Submit jobs with `sbatch`

Submit each script individually from the project root. You can run all three in parallel or sequentially — each writes to its own output directory and log file.

```bash
cd /scratch/<your_username>/VisDrone_YOLO

# End-to-end training (all layers unfrozen)
sbatch Scripts/run_frcnn_end_to_end.sh

# Pretrained backbone baseline (no custom head)
sbatch Scripts/run_frcnn_backbone_only.sh

# Frozen backbone (only DNN head trains)
sbatch Scripts/run_frcnn_frozen_backbone.sh
```

---

### 8. Monitor your jobs

```bash
# Check status of all your jobs
squeue -u <your_username>

# Stream live output from a running job
tail -f /scratch/<your_username>/VisDrone_YOLO/logs/<job_id>.out

# Cancel a job
scancel <job_id>
```

Logs are written to `/scratch/<your_username>/VisDrone_YOLO/logs/` as `<job_id>.out` and `<job_id>.err`.

---

### 9. Outputs and backup

Checkpoints and metrics for each run land in their respective output subdirectory:

```
outputs/frcnn_end_to_end/
outputs/frcnn_backbone_only/
outputs/frcnn_frozen_backbone/
```

> ⚠️ **Scratch space is not backed up and is purged periodically.** Copy important results to your home directory as soon as training completes:
>
> ```bash
> cp -r /scratch/<your_username>/VisDrone_YOLO/outputs ~/VisDrone_outputs_backup
> ```

---

### SLURM resource reference

The `.sh` scripts request the following on the `gpu` partition:

```bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
```

Adjust `--time` and `--mem` as needed. Check available GPU types with `sinfo -p gpu`.

---

### Troubleshooting (OOD cluster)

- **`ModuleNotFoundError`** — ensure `source activate visdrone` is called inside the `.sh` script before the `python` line. If the environment doesn't exist yet, run `bash ~/visdrone/setup.sh` first.
- **`CUDA out of memory`** — reduce the batch size passed to `ResNet50_frcnn_DNN.py`, or request a larger GPU with `--gres=gpu:a100:1`.
- **Job stuck in pending** — check queue depth with `sinfo -p gpu` and `squeue -p gpu`. Peak hours on the `gpu` partition can mean long wait times.
- **`Permission denied` on `.sh` files** — run `chmod +x Scripts/*.sh` (Step 6 above).
- **Scratch quota exceeded** — clean up old runs: `rm -rf /scratch/<your_username>/VisDrone_YOLO/outputs/<old_run>/`.
