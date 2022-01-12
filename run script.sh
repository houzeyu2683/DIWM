. $HOME/.conda/etc/profile.d/conda.sh
environment=kaggle
conda activate $environment
PYTHONPATH='./' python script/run.py
conda deactivate


#echo 'start'
#/bin/bash activate kaggle
#
# source conda activate kaggle
# PYTHONPATH='./' python script/run.py




