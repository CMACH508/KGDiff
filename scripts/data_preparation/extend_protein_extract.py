
import os
import sys
sys.path.append(os.path.abspath('./'))
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial

from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file
from scripts.data_preparation.extract_pockets import process_item
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/extended_proteins')
    parser.add_argument('--dest', type=str, default='./data/extended_poc_proteins')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    
    protein_fn = glob.glob(f'{args.source}/*/*.pdb')
    protein_fn = [os.path.relpath(fn, args.source) for fn in protein_fn]
    molecule_fn = glob.glob(f'{args.source}/*/*.sdf')
    molecule_fn = [os.path.relpath(fn, args.source) for fn in molecule_fn]
    
    index = list(zip(protein_fn, molecule_fn, [0.0]*len(protein_fn), [i for i in range(len(protein_fn))]))

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        index_pocket.append(item_pocket)
    # index_pocket = pool.map(partial(process_item, args=args), index)
    pool.close()

    index_path = os.path.join(args.dest, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index_pocket, f)

    print('Done. %d protein-ligand pairs in total.' % len(index))
    