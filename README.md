# DynaMimicGen
DynaMimicGen: A Data Generation Framework for Robot Learning of Dynamic Tasks
# Abstract
Learning robust manipulation policies typically requires large and diverse datasets, the collection of which is time-consuming, labor-intensive, and often impractical for dynamic environments. In this work, we introduce DynaMimicGen (D-MG), a scalable dataset generation framework that enables policy training from minimal human supervision while uniquely supporting dynamic task settings. Given only a few human demonstrations, D-MG first segments the demonstrations into meaningful sub-tasks, then leverages Dynamic Movement Primitives (DMPs) to adapt and generalize the demonstrated behaviors to novel and dynamically changing environments. Improving prior methods that rely on static assumptions or simplistic trajectory interpolation, D-MG produces smooth, realistic, and task-consistent Cartesian trajectories that adapt in real time to changes in object poses, robot states, or scene geometry during task execution. Our method supports a broad range of scenarios — including scene layouts, object instances, and robot configurations — making it suitable for both static and highly dynamic manipulation tasks. We show that robot agents trained via imitation learning on D-MG generated data achieve strong performance across long-horizon and contact-rich benchmarks, including tasks like cube stacking and placing mugs in drawers, even under unpredictable environment changes. By eliminating the need for extensive human demonstrations and enabling generalization in dynamic settings, D-MG offers a powerful and efficient alternative to manual data collection, paving the way toward scalable, autonomous robot learning.

![DynaMimicGen Logo](dmg_logo.png)

# Spacemouse instructions
This repository is intended to be used with a 3dconnexion spacemouse to record the demonstration
To install the spacemouse dependencies, run:

```bash
pip install pyspacemouse
pip install hid
sudo apt-get install libhidapi-dev
```
Each time a new terminal is started, run the following command:
```bash
sudo chmod 777 /dev/hidraw*
```

# venv installation steps
```bash
cd DynaMimicGen 
pip install -e .
```

# PyTorch & torchvsion installation with CUDA 12.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

# Install [robomimic](https://robomimic.github.io/)
```bash
cd DynaMimicGen
git clone https://github.com/ARISE-Initiative/robomimic.git
git checkout v0.5
cd robomimic/
pip install -e .
```

# Install [mimicgen](https://mimicgen.github.io/)
```bash
cd DynaMimicGen
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen/
pip install -e .
```

# Install [robosuite-task-zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo)
```bash
cd DynaMimicGen
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git
cd robosuite-task-zoo
pip install -e .
```

# Playback demonstration
```bash
python dmg/scripts/playback_human_demonstrations.py --directory <dir-name>
```

# Playback generated trajectories
```bash
python dmg/scripts/playback_human_demonstrations.py --directory <dir-name> --play-dmp
```

# Generate a new dataset with the provided demonstrations:
Substitute <dir-name> with one of the followings (which are the name of the folders in dmg/demos/hdf5)
```bash
python dmg/scripts/generate_dataset.py --directory <dir-name> --num-dmp 10 --render # Generate a dataset of 10 trajectories with scene rendering
```
The previous command will create a folder within the <dir-name> called dmp, in which a demo.hdf5 file will store all the generated trajectories.

# Record a new trajectory
Alternatively, you can record a new trajectory on your own (more treaky procedure):
```bash
python dmg/scripts/collect_human_demonstations.py --environment <env-name> # Collect human demo with space mouse
source dmg/bash/prepare_dataset.sh <dir-name> # Convert to robomimic
python dmg/scripts/annotate_subtasks.py -directory <dir-name> # Annotate subtasks manually
```

# Normalize generated dataset

```bash
python dmg/scripts/normalize_dataset.py --directory StackThree_D0
```

# Run training with Robomimic's Behavior Cloning
```bash
python robomimic/robomimic/scripts/train.py --config dmg/demos/hdf5/<dir-name>/configs/low_dim/bc_rnn.json --abs-actions
```

# Evaluate trained agent
```bash
source dmg/bash/run_trained_agents.sh <dir-name> <type> <ckpt-name> # StackThree_D0_new low_dim 1600
```