# Quantifying VOI of trajectories
Code for paper `Quantifying Intrinsic Value of Information of Trajectories`.

## Getting Started 
1. Dependency
  - Python 3.8
  - PyYAML
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - pandas
  - pytz
  - sklearn
  - gpflow==2.1.4
  - tensorflow

2.	Installation process
  - Upgrade pip: `pip install --upgrade pip`
  - Upgrade setuptools: `pip install --upgrade setuptools`
  - Install dependency: `pip install -r requirements.txt`
  - Create logs directory: e.g. `mkdir logs`
  - Create output directory: e.g. `mkdir -p output/current/`

For GPU installation, uncomment tensorflow-gpu line in `requirements.txt` and/or see `INSTALL.md`.

3. Dataset
Add/Point to GeoLife data in `config.yml`.

## Usage
- Use the `run.py` to run the program: `python run.py --task <task_name> --component <component_name>`.
  - Other parameters are in `config.yml` file.
  
- Before we can quantify the VOI, we need to train and predict the reconstruction models:
  - Train: `python run.py --task benchmark --component reconstruction_training`
  - Predict: `python run.py --task benchmark --component reconstruction_predict`
  
- Then we can quantify VOI with: `python run.py --task benchmark --component quantify_voi`


## Notes
- There was memory leak with Tensorflow (or GPFlow) when we created, trained, and saved models.
We use GPFlow because it allows us to set variances as we wanted.
So we created a new process for each training/predicting task.
See: https://stackoverflow.com/questions/61264461/tensorflow-gpu-not-available-in-a-new-process-forked-off-with-python-2-7s-multi

# Contribute
- Kien Nguyen (duykienvp@gmail.com)
