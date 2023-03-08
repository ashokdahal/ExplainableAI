#!/bin/bash -l

#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=16GB 
#SBATCH --time=10:00:00 
#SBATCH --partition=batch 

source activate dlashok

# get tunneling info
export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}
port=8890

echo $node pinned to port $port
# print tunneling instructions jupyter-log
echo -e "
To connect to the compute node ${node} on IBEX running your jupyter notebook server,
you need to run following two commands in a terminal

1. Command to create ssh tunnel from you workstation/laptop to cs-login:
ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop
"
# Run Jupyter
jupyter notebook --no-browser --port=${port} --ip=${node} 

