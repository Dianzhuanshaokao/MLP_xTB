# Adaptive MLP/xTB Multiscale Simulation for Li–P–S–Cl Solid Electrolyte Interfaces
# 面向 Li–P–S–Cl 固态电解质界面反应的自适应多尺度模拟
## Overview / 项目概述
This project implements an adaptive multiscale simulation framework to study chemical reactions at the interface between lithium metal and Li–P–S–Cl solid electrolytes (e.g., Li<sub>6</sub>PS<sub>8</sub>Cl). The core idea is:
> Use a fast **machine-learned potential (MLP)** for long-timescale molecular dynamics (ML-MD), and dynamically switch to **high-accuracy extended Tight-Binding (xTB)** when rare reaction events (e.g., bond breaking, decomposition) are detected. Collected high-fidelity data are used to iteratively refine the MLP via active learning. 

本项目实现了一个自适应多尺度模拟框架，用于研究金属锂与 Li–P–S–Cl 固态电解质（如 Li<sub>6</sub>PS<sub>8</sub>Cl）界面处的化学反应。其核心思想是：
> 使用快速的 **机器学习势（MLP）** 驱动长时间分子动力学（ML-MD），当检测到稀有反应事件（如键断裂、分解）时，动态切换至高精度 **扩展紧束缚模型 (xTB)** 方法进行局部精算。
所收集的高保真数据通过主动学习迭代优化 MLP 模型
## Key Features / 核心特性
| FEATURE | DESCRIPTION |
|--------|--------|
| Energy-Aligned Hybrid Evaluation  | MLP 与 xTB 能量零点对齐，避免尺度不匹配  | 
| Smooth Force Blending  | 在反应区周围使用距离加权平滑混合力场，防止 MD 轨迹崩溃  | 
| Reaction Event Detection  | 基于键长、配位数、原子逃逸等判据自动识别界面反应  |
| Active Learning Loop  | 自动收集高误差构型，用于微调 MACE 模型 | 
| ASE-Compatible  | 完全兼容 ASE 生态，可直接用于优化、MD、NEB 等任务 |
## System Dependencies / 系统依赖
```bash
# Install xTB (required for xtb-python)
conda install -c conda-forge xtb
```
建议使用 Conda 环境：
```bash
conda create -n multiscale python=3.10
conda activate multiscale
pip install -r requirements.txt
```
## Quick Start / 快速开始
### 1.Clone the repository
```bash
git clone https://github.com/Dianzhuanshaokao/MLP_xTB.git
cd MLP_xTB
```
### 2.Prepare your interface structure
Place your Li/Li<sub>6</sub>PS<sub>8</sub>Cl interface in structures/interface.xyz.

### 3.Run the adaptive simulation
Open `demo.ipynb` in Jupyter and execute cells sequentially:

- Cell 1–4: Environment setup & structure loading
- Cell 5: Adaptive ML-MD with reaction detection
- Cell 6: Fine-tune MACE when enough samples are collected
```
├── demo.ipynb    # Main Jupyter notebook
├── structures/
│   └── interface.xyz                # Your Li/LiPSCl interface model
├── src/
│   ├── Hybrid_calculator.py         # HybridCalculator implementation
│   └── reaction_detector.py         # Unified reaction analysis
├── outputs/
│   ├── adaptive_mlmd.traj           # ML-MD trajectory
│   ├── active_learning_pool.pkl     # High-value configurations
│   └── mace_finetuned.model         # Refined MLP (after fine-tuning)
├── README.md
└── requirments.txt                  # Python dependencies
```
## Example Workflow / 示例工作流
```python
# 1. Initialize calculators
from src.hybrid_calc import HybridCalculator
hybrid_calc = HybridCalculator(mace_calc, xtb_calc)
hybrid_calc.set_energy_offset(initial_atoms)

# 2. Detect reaction
is_reactive, reactive_indices = analyze_reaction(atoms)

# 3. If reactive, evaluate with hybrid calculator
if is_reactive:
    hybrid_calc.calculate(atoms=atoms, reactive_indices=reactive_indices)
    forces = hybrid_calc.results['forces']  # Smooth, stable forces
```
## References / 参考文献
The `MACE` (Machine-learned force fields) architecture, implemented in PyTorch, uses the e3nn library. 
- The MACE training and evaluation code is released under the MIT licence via GitHub and is available at the following link: [https://github.com/ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations)

General Reference to and the implemented GFN methods: `xtb`
- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, S. Spicher, S. Grimme WIREs Comput. Mol. Sci., 2020, 11, e01493. DOI: [10.1002/wcms.1493](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493)

for GFN-xTB:
- S. Grimme, C. Bannwarth, P. Shushkov, J. Chem. Theory Comput., 2017, 13, 1989-2009. DOI: [10.1021/acs.jctc.7b00118](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00118)
- C. Bannwarth, S. Ehlert and S. Grimme., J. Chem. Theory Comput., 2019, 15, 1652-1671. DOI: [10.1021/acs.jctc.8b01176](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176)
- P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv, 2019, preprint. DOI: [10.26434/chemrxiv.8326202.v1](https://chemrxiv.org/engage/chemrxiv/article-details/60c742abbdbb890c7ba3851a)