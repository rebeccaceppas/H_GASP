#!/bin/bash

#SBATCH --account=rrg-kmsmith
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=2
#SBATCH --mem=257000M
#SBATCH --time=1:00:00
#SBATCH --job-name=template_job
#SBATCH --output=/path/to/output_file.out



## loading the modules we need
module --force purge
module use /project/rrg-kmsmith/shared/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2023.06


## running the python script you have created with the pipeline steps
python python_script.py  # for example, full_run.py from tutorials