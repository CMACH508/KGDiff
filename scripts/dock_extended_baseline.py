import argparse
import os
import sys
sys.path.append(os.path.abspath('./'))
import torch
from tqdm.auto import tqdm

from utils.evaluation import scoring_func
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
import multiprocessing as mp
import  pickle
from rdkit import Chem

def dock_pocket_samples(index):
    _, ligand_fn, protein_fn, _ = index
    print('Start docking pocket: %s' % ligand_fn)
    mol = Chem.SDMolSupplier(os.path.join(args.protein_root,ligand_fn))[0]
    pocket_results = []
    try:
        chem_results = scoring_func.get_chem(mol)
        if args.docking_mode == 'qvina':
            vina_task = QVinaDockingTask.from_generated_mol(
                mol, protein_fn, protein_root=args.protein_root, size_factor=args.dock_size_factor)
            vina_results = vina_task.run_sync()
        elif args.docking_mode == 'vina_score':
            vina_task = VinaDockingTask.from_generated_mol(
                mol, protein_fn, protein_root=args.protein_root)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
            docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
            
            vina_results = {
                'score_only': score_only_results,
                'minimize': minimize_results,
                'dock': docking_results
            }
        else:
            raise ValueError
    except:
        print('Error at %s' % ( ligand_fn))
        vina_results = None
    pocket_results.append({
        'lig_mol': mol,
        'protein_fn': protein_fn,
        'chem_results': chem_results,
        'vina': vina_results
        })
    return pocket_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, default=None)
    parser.add_argument('-n', '--num_processes', type=int, default=1)
    parser.add_argument('--protein_root', type=str, default='./data/extended_poc_proteins/')
    parser.add_argument('--dock_size_factor', type=float, default=None)
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--docking_mode', type=str, default='vina_score',
                        choices=['none', 'qvina', 'vina_score'])
    args = parser.parse_args()

    with open(os.path.join(args.protein_root, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)
    with mp.Pool(args.num_processes) as p:
        docked_samples = p.map(dock_pocket_samples, index)
    if args.out is None:
        out_path = os.path.join(args.protein_root, 'testset_docked.pt')
    else:
        out_path = args.out
    torch.save(docked_samples, out_path)
