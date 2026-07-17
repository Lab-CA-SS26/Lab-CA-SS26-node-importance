import argparse
import subprocess
import time
import os
import shutil
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results import append_csv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

experiment_name = "abcde_run"
exp_dir = f"experiments/{experiment_name}_{args.seed}"

print(f"--- Running ABCDE | Seed {args.seed} ---")
print(f"Output dir: {exp_dir}")

if os.path.exists(exp_dir):
    print(f"Removing existing experiment dir: {exp_dir}")
    shutil.rmtree(exp_dir)

start_time = time.time()
train_cmd = ["python", "-u", "baselines/abcde/abcde/train.py", "--seed", str(args.seed), "--name", experiment_name]

os.makedirs("logs", exist_ok=True)
log_file = f"logs/abcde_train_S{args.seed}.log"

try:
    with open(log_file, "w") as out:
        subprocess.check_call(train_cmd, stdout=out, stderr=subprocess.STDOUT)
except subprocess.CalledProcessError:
    print(f"Training failed. Logs in {log_file}")
    sys.exit(1)

training_time = time.time() - start_time
print(f"Training finished in {training_time:.2f}s")

ckpt_dir = os.path.join(exp_dir, "models")
if not os.path.exists(ckpt_dir):
    print(f"Error: Checkpoint directory not found at {ckpt_dir}")
    sys.exit(1)

checkpoints = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
if not checkpoints:
    print(f"Error: No checkpoints in {ckpt_dir}")
    sys.exit(1)

try:
    best_ckpt = sorted(checkpoints,
        key=lambda x: float(re.search(r"val_kendal=([0-9\.\-]+)", x).group(1)), reverse=True)[0]
except (AttributeError, ValueError):
    print("Warning: Could not parse score. Using newest.")
    best_ckpt = max(checkpoints, key=os.path.getmtime)

print(f"Evaluating checkpoint: {best_ckpt}")

algo_name = f"ABCDE_Train_S{args.seed}"
eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_abcde.py")
eval_cmd = ["python", "-u", eval_script, "--model_path", best_ckpt, "--algorithm_name", algo_name]
subprocess.check_call(eval_cmd)

append_csv("all_results_training_time.csv", "Algorithm,Time", f"{algo_name},{training_time:.4f}")
print("Done.")