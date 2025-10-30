#!/bin/bash
#SBATCH --job-name=lithium_battery_sim
#SBATCH --output=output_%j.log
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --partition=CPUcompute

# 加载必要的模块
module load python/3.9

# 激活虚拟环境
source /path/to/your/env/bin/activate

# 运行 Python 脚本
python demo2.py