# Installation guide
## Conda with GPU
This guide is for conda environment. 
It should still work with a normal python environment though.
- Create conda environment: 
  - `conda create --name pup-traj-node4 python=3.8`
- Activate conda environment:
  - `conda activate pup-traj-node4`
- Install Cuda toolkit:
  - `conda install cudatoolkit=11.0`. 
  - This installed the latest cudatoolkit. 
    You should choose the same version as your host machine version 
    (check using `nvidia-smi` command)
  - tensorflow==2.4.1 needs CUDA 11, but if it not available on your system, 
    you may need to build tensorflow from source with CUDA support
- Upgrade pip: `pip3 install pip --upgrade`. Tested with v21.0

- `pip3 install --upgrade-strategy only-if-needed -r requirements.txt`. This used tensorflow-gpu==2.4.1. This version was built targetted cuda 11
    + After this step, if tensorflow does not load GPUs because some libraries not found (e.g. `libcudnn.so.8`), there is potentially a version conflict.
  
- Run: `python run.py --task <task_name> --component <component_name>`

- Check if tensorflow can find GPUs: 
  ```
  import tensorflow as tf
  print(tf.config.list_physical_devices('GPU'))
  ```
  
  - Tensorflow will not see GPUs if any of the required library with correct version missing. 
    If this happened, install the library with correct version
    
  - If there is no GPU detect, the code will still run with CPUs.
  
## Conda with tensorflow-GPU from source
- Follow above to install cudatoolkit in a conda env
  
- Official guide: https://www.tensorflow.org/install/source
- There is also this guide: https://gist.github.com/philwo/f3a8144e46168f23e40f291ffe92e63c
- Install correct cudatoolkit, cudatoolkit-dev, cuddn version from conda and conda-forge 
- Download CUPTI headers/lib from Nvidia page if needed, and extract to folder inside Anaconda environment, e.g., `/home/users/kiennguy/anaconda3/envs/pup-traj-node0/`
- Provide path to CUDA and CUPTI: e.g, `/home/users/kiennguy/anaconda3/envs/pup-traj-node0/`
