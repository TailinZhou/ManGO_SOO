<<<<<<< HEAD
# Learning Design-Score Manifold to Guide Diffusion Models for Offline Optimization

Optimizing complex systems—from discovering therapeutic drugs to designing high-performance materials—remains a fundamental challenge across science and engineering, as the underlying rules are often unknown and costly to evaluate. 
Offline optimization aims to optimize designs for target scores using pre-collected datasets without system interaction.
However, conventional approaches may fail beyond training data, predicting inaccurate scores and generating inferior designs. 
This paper introduces ManGO, a diffusion-based framework that learns the design-score manifold, capturing the design-score interdependencies holistically.
Unlike existing methods that treat design and score spaces in isolation, ManGO unifies forward prediction and backward generation, attaining generalization beyond training data. 
Key to this is its derivative-free guidance for conditional generation, coupled with adaptive inference-time scaling that dynamically optimizes denoising paths. 
Extensive evaluations demonstrate that ManGO outperforms 24 single- and 10 multi-objective optimization methods across diverse domains, including synthetic tasks, robot control, material design, DNA sequence, and real-world engineering optimization.

## Installation

To install and run our code, first clone the `ManGO_SOO` repository.

```
cd ~
git clone https://github.com/TailinZhou/ManGO_SOO.git
cd ManGO_SOO
```

Update your conda environment to include the necessary libraries. if sudo:
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
if you are not using sudo, you can use the following command to add the necessary libraries (some errors may occur):
```
conda install -c conda-forge mesalib glew glfw
```

Next, create a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from the `environment.yml` file to setup the environment and install the relevant dependencies.

```
conda env create -f environment.yml
conda activate mango4soo
pip install -r mango_requirements.txt
```

For people who installed these libraries to the non-base conda environment and have to batch script on a cluster machine, you might want to add this to your .bashrc. Note how CPATH comes after conda activate
```
# export CPATH=$CONDA_PREFIX/include
# conda activate mango4soo
# export CPATH=$CONDA_PREFIX/include
```

For mujoco200，if you already have mujoco200 installed, you can skip this step.

```
mkdir ~/.mujoco
cp mujoco200_linux.zip ~/.mujoco
cd ~/.mujoco
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200   
```
Additionally, you need to download the `mjkey.txt` file from [here](https://www.roboti.us/license.html) and place it in the `~/.mujoco` directory.
Add the license key for mujoco200 to the `mjkey.txt` file.
```
cd ManGO_SOO
cp mjkey.txt ~/.mujoco
cp mjkey.txt ~/.mujoco/mujoco200/bin
```

set the `LD_LIBRARY_PATH` environment variable to point to the `mujoco200/bin` directory.
```
conda env config vars set -n mango4soo LD_LIBRARY_PATH="$LD_LIBRARY_PATH: {$HOME}/.mujoco/mujoco200/bin"
```
please replace `$HOME` with your home directory path (you can use echo $HOME to get your home directory path).
 

Then, since we may use minmax normalization for training ManGO, instead of zero_unit_variance normalization in the original benchmark setting.  Specifically, zero_unit_variance for superconduct, min_max for other tasks, due to a bug of overestimating evaluation in superconduct if using minmax normalization. Please replace our `design_bench` directory with the `design_bench` package in the mango4soo environment, please replace the following `~/your_anaconda3_path` as your anaconda path.

```
TARGET_DIR="~/your_anaconda3_path/envs/mango4soo/lib/python3.8/site-packages/design_bench"
 ls $TARGET_DIR   
 SOURCE_DIR="~/ManGO_SOO/design_bench"
ln -s "$SOURCE_DIR" "$TARGET_DIR"  
```
<!-- cp -r "$SOURCE_DIR" "$TARGET_DIR" 直接复制覆盖​ -->

e.g.,
```
TARGET_DIR="/home/tzhouaq/anaconda3/envs/mango4soo/lib/python3.8/site-packages/design_bench"
ls $TARGET_DIR
SOURCE_DIR="~/ManGO_SOO/design_bench"
ln -s "$SOURCE_DIR" "$TARGET_DIR"  
```
 

Next, please also follow the directions [here](https://github.com/rail-berkeley/design-bench/issues/1) to download the `design-bench`-associated datasets and save them to the `~/your_anaconda3_path/envs/mango4soo/lib/python3.8/site-packages/design_bench_data` package directory, and also copy the [`smiles_vocab.txt`](./data/molecules/smiles_vocab.txt) file to the `design_bench_data` package directory:

```
cd ~/ManGO_SOO
mkdir -p "~/your_anaconda3_path/envs/mango4soo/lib/python3.8/site-packages/design_bench_data/"
cp -p data/molecules/smiles_vocab.txt  ~/your_anaconda3_path/envs/mango4soo/lib/python3.8/site-packages/design_bench_data/
```

```
e.g.,:
cd ~/ManGO_SOO
mkdir -p "/home/tzhouaq/anaconda3/envs/mango4soo/lib/python3.8/site-packages/design_bench_data/"
cp -p data/molecules/smiles_vocab.txt /home/tzhouaq/anaconda3/envs/mango4soo/lib/python3.8/site-packages/design_bench_data/
```

### Running
After successful installation, you can run our jupyter scripts in the `mango_jupyter_scripts` directory, where we provide our pretrained models and the corresponding evaluation results.


## Contact
Questions and comments are welcome. Suggestions can be submitted through Github issues. 

## Citation
@inproceedings{mango,
    author = {Tailin Zhou, Zhilin Chen, Wenlong Lyv, Zhitang Chen, Danny H.K. Tsang, and Jun Zhang.},
    title = {Learning Design-Score Manifold to Guide Diffusion Models for Offline Optimization},
    booktitle = {under review},
    year = {2025},
}

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
=======
# ManGO_SOO
Codes for ManGO on offline SOO tasks via design benchmark
>>>>>>> 12eeb0f022e09171357231b6cd171a7b46ed6a8d
