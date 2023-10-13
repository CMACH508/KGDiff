<!--
 * @Author: QHGG
 * @Date: 2023-10-08 16:50:32
 * @LastEditTime: 2023-10-13 23:00:27
 * @LastEditors: QHGG
 * @Description: 
 * @FilePath: /KGDiff/README.md
-->
# KGDiff: Towards Explainable Target-Aware Molecule Generation with Knowledge Guidance 
<a href="https://github.com/CMACH508/KGDiff/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/autoreviewer" />
    </a>
<a href="https://doi.org/10.5281/zenodo.8419944"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8419944.svg" alt="DOI"></a>


## 🚀 Installation
Before running KGDiff, please follow the below instruction to build the virtualenv.

```bash
conda create -n kgdiff python=3.9
conda activate kgdiff
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
conda install pytorch-scatter pytorch-cluster pytorch-sparse==0.6.13 pyg==2.0.4 -c pyg
pip install pyyaml easydict lmdb
pip install numpy==1.21.6 pandas==1.4.1 tensorboard==2.9.0 seaborn==0.11.2 
pip install Pillow==9.0.1
conda install -c conda-forge openbabel
pip install meeko==0.1.dev3 vina==1.2.2 pdb2pqr rdkit

# =======================
# install autodocktools
# for linux
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
# for windows
python.exe -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
# =======================
pip install scipy==1.7.3

# be cautious with package version, feel free to open an issue if you meet package conflits.
```

## 💾 Datasets
We have uploaded the datasets (CrossDocked2020 and PDBBind2020) to [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8419944.svg)](https://doi.org/10.5281/zenodo.8419944). Prior to training and generation, kindly download and extract the **data.zip(datasets)** to the project's root directory.

## Pretrained Models

We have uploaded the pretrained models to **logs_diffusion.zip(model ckpts)** on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8419944.svg)](https://doi.org/10.5281/zenodo.8419944). Feel free to download and place them in the root directory.

## 🛠️ Training KGDiff from Scratch

We recommend training the model on a single GPU with 40GB CUDA memory. The training parameters can be found in the configs/training.yml file.


```bash
git clone https://github.com/CMACH508/KGDiff.git
cd KGDiff
python scripts/train_diffusion.py
```

## Generating Molecules

Here, we provide two examples for molecule generation.


-   To sample molecules based on the first protein from the test set in CrossDocked2020, run the following scripts. If you want to sample from PDBBind2020, replace the arguments "--guide_mode joint" with "--guide_mode pdbbind_random.
    ```bash
    python scripts/sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode joint --type_grad_weight 100 --pos_grad_weight 25 --result_path ./cd2020_pro_0_res
    ```


-   To sample molecules based on the proteins listed in Table S4 of our paper, run the following scripts.
    ```bash
    python scripts/sample_for_pocket.py --pdb_idx 0 --protein_root ./data/extended_poc_proteins/  --guide_mode joint --type_grad_weight 100 --pos_grad_weight 25 --result_path ./extended_pro_0_res
    ```

## Evaluating Generated Molecules
Here, we offer an example for evaluating generated molecules.

```bash
python scripts/evaluate_diffusion.py
```

## Installing and Running KGDiff from PyPI

*note: Before you install KGDiff, please create an virtual env in [Installation](#-installation) part.*

| command  | excuting files | 
|:------:|:------:|
| kg_gen    | scripts.sample_diffusion.py     | 
| kg_gen4poc    | scripts.sample_for_pocket.py     | 
| kg_train    | scripts.train_diffusion.py     | 
| kg_eval    | scripts.evaluate_diffusion.py     | 

Here is an example for training KGDiff.
```bash
conda activate kgdiff
pip install KGDiff==0.1.2
kg_train --config your_config_path --ckpt your_ckpt_path --logdir your_ckpt_dirname
# Detailed arguments are given in scripts/train_diffusion.py
```

## Reproducing Our Paper
We provide the **reproduction.ipynb** notebook file for reproducing figures and benchmarks in our paper. Before proceeding, please ensure you have downloaded and extracted the **misc_results.zip** and the **benchmark.zip** from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8419944.svg)](https://doi.org/10.5281/zenodo.8419944) into the root directory.

## ⚖️ License

The code in this package is licensed under the MIT License. We thanks TargetDiff for the open source codes.

## ✉️ Contact
If you have any question, please contact us: qhonearth@sjtu.edu.cn.