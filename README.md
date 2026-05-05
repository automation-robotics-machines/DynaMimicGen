<p align="center">
  <img src="dmg_logo.png" width="1000" alt="DynaMimicGen Logo">
</p>

<h1 align="center">DynaMimicGen (D-MG)</h1>
<h3 align="center">A Data Generation Framework for Robot Learning of Dynamic Tasks</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2511.16223">
    <img src="https://img.shields.io/badge/arXiv-2511.16223-b31b1b?logo=arxiv&logoColor=white" height="28">
  </a>
  <a href="https://github.com/automation-robotics-machines/DynaMimicGen">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" height="28">
  </a>
</p>

---

## Abstract

Learning robust manipulation policies typically requires large and diverse datasets, the collection of which is time-consuming, labor-intensive, and often impractical for dynamic environments. **DynaMimicGen (D-MG)** is a scalable dataset generation framework that enables policy training from minimal human supervision while uniquely supporting dynamic task settings.

Given only a few human demonstrations, D-MG:
1. Segments demonstrations into meaningful sub-tasks
2. Leverages **Dynamic Movement Primitives (DMPs)** to adapt behaviors to novel, dynamically changing environments
3. Produces smooth, realistic, and task-consistent **Cartesian trajectories** that adapt in real time to changes in object poses, robot states, or scene geometry

Agents trained via imitation learning on D-MG-generated data achieve strong performance on long-horizon and contact-rich benchmarks — even under unpredictable environment changes.

---

## Requirements

- Python 3.8+
- CUDA 12.8 (for GPU training)
- A [3DConnexion SpaceMouse](https://3dconnexion.com/) (for recording new demonstrations)

---

## Installation

### 1. Clone & Install D-MG

```bash
git clone https://github.com/automation-robotics-machines/DynaMimicGen.git
cd DynaMimicGen
pip install -e .
```

### 2. PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Dependencies

**[robomimic](https://robomimic.github.io/)**
```bash
cd DynaMimicGen
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic && git checkout v0.5
pip install -e .
```

**[MimicGen](https://mimicgen.github.io/)**
```bash
cd DynaMimicGen
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen && pip install -e .
```

**[robosuite-task-zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo)**
```bash
cd DynaMimicGen
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git
cd robosuite-task-zoo && pip install -e .
```

### 4. SpaceMouse Setup

Install drivers for recording human demonstrations:

```bash
pip install pyspacemouse hid
sudo apt-get install libhidapi-dev
```

> ⚠️ **Every time you open a new terminal**, run:
> ```bash
> sudo chmod 777 /dev/hidraw*
> ```

---

## Usage

### Playback

```bash
# Playback human demonstrations
python dmg/scripts/playback_human_demonstrations.py --directory <dir-name>

# Playback DMP-generated trajectories
python dmg/scripts/playback_human_demonstrations.py --directory <dir-name> --play-dmp
```

---

### Generate a Dataset

Use one of the provided demo folders under `dmg/demos/hdf5/` as `<dir-name>`:

```bash
python dmg/scripts/generate_dataset.py \
  --directory <dir-name> \
  --num-dmp 10 \
  --render
```

This creates a `dmp/demo.hdf5` file inside `<dir-name>` containing all generated trajectories.

Then extract observations:

```bash
python robomimic/robomimic/scripts/conversion/convert_robosuite.py \
  --dataset dmg/demos/hdf5/<dir-name>/dmp/demo.hdf5

python robomimic/robomimic/scripts/dataset_states_to_obs.py \
  --dataset dmg/demos/hdf5/<dir-name>/dmp/demo.hdf5 \
  --output_name image.hdf5 \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand \
  --camera_height 84 --camera_width 84 \
  --exclude-next-obs
```

---

### Record a New Demonstration

> ⚠️ This requires a SpaceMouse and is more involved than using the provided demos.

```bash
# Step 1 — Collect human demonstration
python dmg/scripts/collect_human_demonstations.py --environment <env-name>

# Step 2 — Convert to robomimic format
python robomimic/robomimic/scripts/conversion/convert_robosuite.py \
  --dataset dmg/demos/hdf5/<dir-name>/demo.hdf5

# Step 3 — Extract observations
python robomimic/robomimic/scripts/dataset_states_to_obs.py \
  --dataset dmg/demos/hdf5/<dir-name>/demo.hdf5 \
  --output_name image.hdf5 \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand \
  --camera_height 84 --camera_width 84 \
  --exclude-next-obs

# Step 4 — Annotate sub-tasks manually
python dmg/scripts/annotate_subtasks.py --directory <dir-name>
```

---

### Train a Policy

**Behavior Cloning (BC-RNN)**

```bash
# Image-based
python robomimic/robomimic/scripts/train.py \
  --config dmg/configs/<task-name>/bc_configs/image/bc_rnn.json

# Low-dimensional
python robomimic/robomimic/scripts/train.py \
  --config dmg/configs/<task-name>/bc_configs/low_dim/bc_rnn.json
```

**Diffusion Policy**

```bash
# Image-based
python robomimic/robomimic/scripts/train.py \
  --config dmg/configs/<task-name>/dp_configs/image/dp.json

# Low-dimensional
python robomimic/robomimic/scripts/train.py \
  --config dmg/configs/<task-name>/dp_configs/low_dim/dp.json
```

---

### Evaluate a Trained Agent

```bash
source dmg/bash/run_trained.sh <dir-name> <type> <ckpt-name> <horizon> <folder_name>
```

**Example:**
```bash
source dmg/bash/run_trained_agents.sh Stack_D0_new low_dim 1600 700 20250723221119
```

---

## Reproducing the Paper Results

Detailed instructions, configurations, and benchmark assets required to fully reproduce the experiments reported in the paper are available upon request.

For reproducibility inquiries, please contact: **Vincenzo Pomponi** — vincenzo.pomponi@supsi.ch
---

## Citation

If you use **DynaMimicGen** in your work, please cite:

```bibtex
@article{pomponi2025dynamimicgen,
  title     = {DynaMimicGen: A Data Generation Framework for Robot Learning of Dynamic Tasks},
  author    = {Pomponi, Vincenzo and Franceschi, Paolo and Baraldo, Stefano and
               Roveda, Loris and Avram, Oliver and Gambardella, Luca Maria and Valente, Anna},
  journal   = {arXiv preprint arXiv:2511.16223},
  year      = {2025}
}
```
