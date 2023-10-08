import argparse
import os
import sys
sys.path.append(os.path.abspath('./'))
import torch
import glob
from tqdm.auto import tqdm
from easydict import EasyDict
from rdkit import Chem
from utils import misc
from datasets import get_dataset
from utils.evaluation.docking_vina import VinaDockingTask



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('-s', '--split', type=str, default='./data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('-o', '--out', type=str, default='./cdock')
    parser.add_argument('--protein_root', type=str, default='./data/test_set/')
    parser.add_argument('--pro_idx', type=int, default=0)
    parser.add_argument('--mol_idx', type=int, default=72, choices=[72,73])  
    parser.add_argument('--exhaustiveness', type=int, default=16)  
    
    args = parser.parse_args()

    logger = misc.get_logger('docking')
    logger.info(args)

    # Load dataset
    dataset, subsets = get_dataset(
        config=EasyDict({
            'name': 'pl',
            'path': args.dataset,
            'split': args.split
        })
    )
    _, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Dock
    logger.info('Start docking with protein %s %s'%(args.pro_idx, test_set[args.pro_idx]['protein_filename']))
    results = []
    pro_info = test_set[args.pro_idx]
    mols4crossdock = glob.glob('./results/joint_pv_type100_pos25_full/eval_results/sdf_%d/*'%(args.mol_idx))
    known_lig = Chem.SDMolSupplier(os.path.join(args.protein_root, pro_info['ligand_filename']))[0]
    known_lig_center = known_lig.GetConformer(0).GetPositions().mean(0)
    for i, data in enumerate(tqdm(mols4crossdock)):
        mol = Chem.SDMolSupplier(data)[0]
        
        # move center of generated molecules to known ligand
        pos = mol.GetConformer(0).GetPositions()
        pos = pos - pos.mean(0) + known_lig_center
        conf = mol.GetConformer(0)
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i])
        assert (mol.GetConformer(0).GetPositions() - pos).sum() < 1e-6
        
        protein_fn = os.path.join(
            os.path.dirname(pro_info['ligand_filename']),
            os.path.basename(pro_info['ligand_filename'])[:10] + '.pdb'
        )
        try:
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
        except:
            print('Error at %d of %s' % (i, pro_info['protein_filename']))
            vina_results = None

        results.append({
            'mol_trans': mol,
            'origin_mol': Chem.SDMolSupplier(data)[0],
            'mol_filename': data,
            'vina': vina_results
        })

    # Save
    out_path = args.out
    os.makedirs(out_path, exist_ok=True)
    logger.info('Num docked: %d' % len(results))
    logger.info('Saving results to %s' % out_path)
    torch.save(results, os.path.join(out_path, 'sdf%s_pro%d.pt'%(args.mol_idx,args.pro_idx)))