import os
import sys
sys.path.append(os.path.abspath('./'))
import argparse
import shutil
from glob import glob
import pickle
import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein
from datasets.pl_pair_dataset import parse_sdf_file

def pdb_to_pocket_data(protein_root, protein_fn, ligand_fn):
    pocket_dict = PDBProtein(os.path.join(protein_root,protein_fn)).to_dict_atom()
    ligand_dict = parse_sdf_file(os.path.join(protein_root, ligand_fn))
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict = torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_idx', type=int, default=0)
    parser.add_argument('--protein_root', type=str, default='./data/extended_poc_proteins/')
    parser.add_argument('--config', type=str, default='./configs/sampling.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--guide_mode', type=str, default='joint', choices=['joint', 'vina', 'valuenet', 'wo'])  
    parser.add_argument('--type_grad_weight', type=float, default=100)
    parser.add_argument('--pos_grad_weight', type=float, default=25)
    parser.add_argument('--result_path', type=str, default='./test_poc')
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    if args.guide_mode == 'joint':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'vina':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'valuenet':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = torch.load(config.model['value_ckpt'], map_location=args.device)
    elif args.guide_mode == 'wo':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    else:
        raise NotImplementedError
    
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")
    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])


    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    
    if value_ckpt is not None:
        # value model
        value_model = ScorePosNet3D(
            value_ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        value_model.load_state_dict(value_ckpt['model'])
    else:
        value_model = None
    
    with open(os.path.join(args.protein_root, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)

    protein_fn, ligand_fn, _, _ = index[args.pdb_idx]
    # Load pocket
    data = pdb_to_pocket_data(args.protein_root, protein_fn, ligand_fn)
    data = transform(data)
    
    pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        guide_mode=args.guide_mode,
        value_model=value_model,
        type_grad_weight=args.type_grad_weight,
        pos_grad_weight=args.pos_grad_weight
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_exp': pred_exp,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'pred_exp_traj': pred_exp_traj,
        'pred_exp_atom_traj': pred_exp_atom_traj,
        'time': time_list
    }
    logger.info('Sample done!')

    
    torch.save(result, os.path.join(result_path, f'result_{os.path.basename(protein_fn)[:-4]}.pt'))


if __name__ == '__main__':
    main()