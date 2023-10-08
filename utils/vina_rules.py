'''
Author: QHGG
Date: 2023-05-06 20:50:46
LastEditTime: 2023-05-08 14:01:50
LastEditors: QHGG
Description: 
FilePath: /IA-drug/utils/vina_rules.py
'''
import torch
from utils import transforms
from torch_scatter import scatter_add

ATOM_VANDER_R = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.35,
    15: 1.90,
    16: 1.85,
    17: 1.80,
    34: 1.90
}

VINA_WEIGHT = [-0.0356, -0.00516, 0.840, -0.0351, -0.587]

def gauss1(d):
    return torch.exp(-(2*d)**2)

def gauss2(d):
    return torch.exp(-((d-3.0)/2)**2)

def repulsion(d):
    return torch.where(d < 0, d**2, torch.zeros_like(d))

def hydrophobic(d):
    return torch.clamp(1.5 - d, min=0.0,max=1.0)

def hbonding(d):
    return torch.clamp(d/(-0.7), min=0.0,max=1.0)
    
def compute_d(batch_atom1, batch_atom2, dis_matrix):
    atom1_vander_radi = torch.tensor([ATOM_VANDER_R[i] for i in batch_atom1]).to(dis_matrix.device)
    atom2_vander_radi = torch.tensor([ATOM_VANDER_R[i] for i in batch_atom2]).to(dis_matrix.device)
    vander_radi_pair = atom1_vander_radi[:, None] + atom2_vander_radi[None, :]
    d_ij = dis_matrix[:,None,:] - vander_radi_pair[None,:,:]
    
    return d_ij

def vina_score(batch_atom1, batch_atom2, dis_matrix):
    
    pass

def calc_vina(log_ligand_v_recon, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein):
    
    # ligand_v_atomic = transforms.get_atomic_number_from_index(log_ligand_v_recon, mode='add_aromatic')
    
    ligand_v_atomic = transforms.get_atomic_number_from_index(torch.arange(0,log_ligand_v_recon.shape[-1]), mode='add_aromatic')
    
    protein_v_atomic = transforms.get_protein_atomic_number_from_index(torch.argmax(protein_v[:,:6],dim=-1))
    
    dis_matrix = torch.sqrt(((ligand_pos[:,None,:] - protein_pos[None,:,:])**2).sum(dim=-1))
    dij = compute_d(ligand_v_atomic, protein_v_atomic, dis_matrix)
    
    dij = (dij * log_ligand_v_recon[:,:,None]).sum(dim=1)
    
    gauss1_score = gauss1(dij)
    gauss2_score = gauss2(dij)
    repulsion_score = repulsion(dij)
    hydrophobic_score = hydrophobic(dij)
    hbonding_score = hbonding(dij)
    
    vina_score = torch.stack((
        gauss1_score, 
        gauss2_score, 
        repulsion_score, 
        hydrophobic_score, 
        hbonding_score
    ),dim=0) * (torch.tensor(VINA_WEIGHT)[:, None, None].to(gauss1_score.device))
    
    dis_mask = dis_matrix < 8.0
    
    vina_score = vina_score * dis_mask[None,:,:]
    vina_score_add = scatter_add(scatter_add(vina_score, batch_protein[None,None,:]), batch_ligand[None,:, None],dim=1)
    
    vina_score_final = vina_score_add.sum(dim=0)
    return torch.diagonal(vina_score_final,dim1=-2,dim2=-1), torch.diagonal(vina_score_add,dim1=-2,dim2=-1)
