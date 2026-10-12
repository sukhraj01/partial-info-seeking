#!/bin/bash
#SBATCH --job-name=hm-decision
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/decision_%j.out
#SBATCH --error=logs/decision_%j.out

# Thompson sampling / epsilon-greedy / cold-start trace on the frozen 57M @ 390K
# model. No training. Runs cache per-run npz files, so a rerun resumes.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
[ -f ada_modules.sh ] && source ./ada_modules.sh
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tests_decision_strategies.py
python run_decision_strategies.py --checkpoint models/transformer_CAP57.pt --batch-size 2048
echo "finished $(date)"
