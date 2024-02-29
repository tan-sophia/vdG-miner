#! /usr/bin/env python3
"""
Zack Mawaldi and Rian Kormos

Converts structures into 3Di sequences.

./foldseek.py encoder.pkl some_pdb.pdb --virt 270 0 2

Cobbled from:
https://github.com/steineggerlab/foldseek-analysis
"""
import os
import sys
import pickle
import argparse

import numpy as np
import prody as pr

from scipy.spatial.distance import cdist


def discretize(weights_0, bias_0, weights_1, bias_1, weights_2, bias_2, 
               centroids, x):
    """Discretize the input data as 3Di sequence tokens using the centroids.

    Parameters
    ----------
    weights_0 : np.ndarray
        The weights for the first MLP layer.
    bias_0 : np.ndarray
        The bias for the first MLP layer.
    weights_1 : np.ndarray
        The weights for the second MLP layer.
    bias_1 : np.ndarray
        The bias for the second MLP layer.
    weights_2 : np.ndarray
        The weights for the linear layer.
    bias_2 : np.ndarray
        The bias for the linear layer.
    centroids : np.ndarray
        The centroids to use for discretization.
    x : np.ndarray
        The input data.
    
    Returns
    -------
    y : np.ndarray
        The discretized data as a 3Di sequence (in integer form).
    """
    out0 = np.clip(bias_0 + np.dot(x, weights_0.T), 0, np.inf) # MLP layer
    out1 = np.clip(bias_1 + np.dot(out0, weights_1.T), 0, np.inf) # MLP layer
    z = bias_2 + np.dot(out1, weights_2.T) # Linear layer
    return np.argmin(cdist(z, centroids, metric='sqeuclidean'), axis=1)


def unit_vec(v):
    """Compute the unit vector in the same direction as v.

    Parameters
    ----------
    v : np.ndarray [N_vec x 3]
        The vectors to normalize.

    Returns
    -------
    u : np.ndarray [N_vec x 3]
        The unit vectors in the same direction as the vectors v.
    """
    return v / np.linalg.norm(v, axis=1, keepdims=1)


def approx_c_beta_position(c_alpha, n, c_carboxyl):
    """Approximate C_beta position from C_alpha, N and C positions.

       Assume the four atoms bonded to the C_alpha form a regular tetrahedron.

    Parameters
    ----------
    c_alpha : np.ndarray [n_res x 3]
        The C_alpha positions.
    n : np.ndarray  [n_res x 3]
        The N positions.
    c_carboxyl : np.ndarray [n_res x 3]
        The C_carboxyl positions.

    Returns
    -------
    c_beta : np.ndarray [n_res x 3]
        The approximate C_beta positions.
    """
    v1 = unit_vec(c_carboxyl - c_alpha)
    v2 = unit_vec(n - c_alpha)

    b1 = v2 + 1/3 * v1
    b2 = np.cross(v1, b1)

    u1 = unit_vec(b1)
    u2 = unit_vec(b2)

    # direction from c_alpha to c_beta
    v4 = -1/3 * v1 + \
         np.sqrt(8)/3 * \
         np.linalg.norm(v1, axis=1, keepdims=True) * \
         (-1/2 * u1 - np.sqrt(3)/2 * u2)
    
    DISTANCE_ALPHA_BETA = 1.5336

    return c_alpha + DISTANCE_ALPHA_BETA * v4  # c_beta


def calc_features(coords, valid_mask):
    """Given protein backbone coordinates, calculate the FoldSeek features.

    Parameters
    ----------
    coords : np.ndarray [n_res x 3]
        The coordinates of the residues.
    valid_mask : np.ndarray [n_res]
        The mask of valid residues.

    Returns
    -------
    features : np.ndarray [n_pairs]
        The FoldSeek features for each pair of residues.
    valid_mask_pairs : np.ndarray [n_pairs]
        The mask of valid pairs.
    """
    n_res = coords.shape[0]
    out = np.full((n_res, 10), np.nan, dtype=np.float32)

    # find the residues i such that i - 1, i, i + 1, j - 1, j, and j + 1
    # are all in valid_mask and the CA-CA distances are appropriate
    CA = coords[:, :3]
    CA_dists = np.linalg.norm(CA[1:] - CA[:-1], axis=1)
    i_good_dist = np.argwhere(CA_dists < 4.).flatten()
    i_good_dist = np.intersect1d(i_good_dist, i_good_dist + 1)

    CB = coords[valid_mask, 3:6]
    i = np.arange(n_res)
    j = -np.ones_like(i)
    j[valid_mask] = np.argmin(cdist(CB, CB) + np.eye(len(CB)) * 1e6, 
                              axis=1).flatten()
    mask = np.in1d(i - 1, valid_mask) & np.in1d(i, valid_mask) & \
           np.in1d(i + 1, valid_mask) & np.in1d(j - 1, valid_mask) & \
           np.in1d(j, valid_mask) & np.in1d(j + 1, valid_mask) & \
           np.in1d(i, i_good_dist) & np.in1d(j, i_good_dist)
    i = i[mask]
    j = j[mask]

    u_1 = unit_vec(CA[i]     - CA[i - 1])
    u_2 = unit_vec(CA[i + 1] - CA[i])
    u_3 = unit_vec(CA[j]     - CA[j - 1])
    u_4 = unit_vec(CA[j + 1] - CA[j])
    u_5 = unit_vec(CA[j]     - CA[i])

    cos_phi_12 = (u_1 * u_2).sum(axis=1)
    cos_phi_34 = (u_3 * u_4).sum(axis=1)
    cos_phi_15 = (u_1 * u_5).sum(axis=1)
    cos_phi_35 = (u_3 * u_5).sum(axis=1)
    cos_phi_14 = (u_1 * u_4).sum(axis=1)
    cos_phi_23 = (u_2 * u_3).sum(axis=1)
    cos_phi_13 = (u_1 * u_3).sum(axis=1)

    d = np.linalg.norm(CA[i] - CA[j], axis=1)
    seq_dist = (j - i).clip(-4, 4)
    log_dist = np.sign(j - i) * np.log(np.abs(j - i) + 1)

    out[mask] = np.vstack([cos_phi_12, cos_phi_34,
                           cos_phi_15, cos_phi_35,
                           cos_phi_14, cos_phi_23,
                           cos_phi_13, d, seq_dist, log_dist]).T
    
    valid_mask_pairs = ~np.isnan(out).any(axis=1)

    return out, valid_mask_pairs


def move_CB(coords, c_alpha_beta_distance_scale=1, virt_cb=None):
    """Move C_beta by scaling the distance to C_alpha or using virtual coords.

    Parameters
    ----------
    coords : np.ndarray [n_res x 6]
        The backbone atomic coordinates of a full protein.
    c_alpha_beta_distance_scale : float
        The scaling factor for the distance between C_alpha and C_beta.
    virt_cb : list
        The virtual C_beta position in the frame of the backbone atoms.

    Returns
    -------
    coords : np.ndarray [n_res x 6]
        The backbone atomic coordinates of a full protein with C_beta moved.
    """
    # replace CB coordinates with position along CA-CB vector
    if c_alpha_beta_distance_scale != 1 and virt_cb is None:
        ca = coords[:, 0:3]
        cb = coords[:, 3:6]
        coords[:, 3:6] = (cb - ca) * c_alpha_beta_distance_scale + ca

    # instead of CB use point defined by two angles and a distance
    if virt_cb is not None:
        alpha, beta, d = virt_cb

        alpha = np.radians(alpha)
        beta = np.radians(beta)

        ca = coords[:, 0:3]
        cb = coords[:, 3:6]
        n_atm = coords[:, 6:9]
        co_atm = coords[:, 9:12]

        v = cb - ca

        # normal angle (between CA-N and CA-VIRT)
        a = cb - ca
        b = n_atm - ca
        k = np.cross(a, b) / np.linalg.norm(np.cross(a, b), axis=1, keepdims=1)

        # Rodrigues rotation formula
        v = v * np.cos(alpha) + np.cross(k, v) * np.sin(alpha) + \
            k * (k * v).sum(axis=1, keepdims=1) * (1 - np.cos(alpha))

        # dihedral angle (axis: CA-N, CO, VIRT)
        k = (n_atm - ca) / np.linalg.norm(n_atm - ca, axis=1, keepdims=1)
        v = v * np.cos(beta) + np.cross(k, v) * np.sin(beta) + \
            k * (k * v).sum(axis=1, keepdims=1) * (1 - np.cos(beta))

        coords[:, 3:6] = ca + v * d
    return coords


def get_coords_from_pdb(path, full_backbone=False):
    """Read pdb file and return CA + CB (+ N + C) coords.
    
       CB from GLY are approximated.

    Parameters
    ----------
    path : str
        The path to the PDB file.
    full_backbone : bool
        Whether to return N and C coordinates as well.

    Returns
    -------
    coords : np.ndarray [n_res x 6 or 12]
        The atomic coordinates.
    valid_mask : np.ndarray [n_res]
        The mask of valid residues.
    """
    if type(path) is str:
        structure = pr.parsePDB(path)
    else:
        structure = path # allow for prody structure as input
    n_res = structure.getResindices().max() + 1

    # coordinates of C_alpha and C_beta
    n_cols = 6 if not full_backbone else 12
    # CA, CB(, N, C)
    coords = np.full((n_res, n_cols), np.nan, dtype=np.float32)

    ca_atoms = structure.select('name CA')
    cb_atoms = structure.select('name CB')
    n_atoms = structure.select('name N')
    c_atoms = structure.select('name C')

    resindices_ca, idxs_ca = np.unique(ca_atoms.getResindices(), 
                                       return_index=True)
    resindices_cb, idxs_cb = np.unique(cb_atoms.getResindices(),
                                       return_index=True)
    resindices_n, idxs_n = np.unique(n_atoms.getResindices(), 
                                     return_index=True)
    resindices_c, idxs_c = np.unique(c_atoms.getResindices(),
                                     return_index=True)

    ca_coords = ca_atoms.getCoords()[idxs_ca]
    cb_coords = cb_atoms.getCoords()[idxs_cb]
    n_coords = n_atoms.getCoords()[idxs_n]
    c_coords = c_atoms.getCoords()[idxs_c]

    # determine masks for indexing coords
    valid_mask = np.intersect1d(resindices_ca, 
                                np.intersect1d(resindices_n, 
                                               resindices_c))
    valid_mask_has_cb = np.intersect1d(valid_mask, resindices_cb)
    valid_mask_non_cb = np.setdiff1d(valid_mask, resindices_cb)

    # determine masks for indexing ca_coords, cb_coords, n_coords & c_coords
    mask_ca = np.argwhere(np.in1d(resindices_ca, valid_mask)).flatten()
    mask_cb = np.argwhere(np.in1d(resindices_cb, valid_mask_has_cb)).flatten()
    mask_ca_non_cb = np.argwhere(np.in1d(resindices_ca, 
                                         valid_mask_non_cb)).flatten()
    mask_n_non_cb = np.argwhere(np.in1d(resindices_n, 
                                        valid_mask_non_cb)).flatten()
    mask_c_non_cb = np.argwhere(np.in1d(resindices_c, 
                                        valid_mask_non_cb)).flatten()
    mask_n = np.argwhere(np.in1d(resindices_n, valid_mask)).flatten()
    mask_c = np.argwhere(np.in1d(resindices_c, valid_mask)).flatten()

    coords[valid_mask, 0:3] = ca_coords[mask_ca]
    coords[valid_mask_has_cb, 3:6] = cb_coords[mask_cb]
    coords[valid_mask_non_cb, 3:6] = approx_c_beta_position(
            ca_coords[mask_ca_non_cb], 
            n_coords[mask_n_non_cb], 
            c_coords[mask_c_non_cb])
    if full_backbone:
        coords[valid_mask, 6:9] = n_coords[mask_n]
        coords[valid_mask, 9:12] = c_coords[mask_c]

    return coords, valid_mask


def encoder_features(pdb_path, virt_cb):
    """Calculate 3D descriptors for each residue of a PDB file.

    Parameters
    ----------
    pdb_path : str
        The path to the PDB file.
    virt_cb : list
        The virtual C_beta position in the frame of the backbone atoms.

    Returns
    -------
    vae_features : np.ndarray [n_res x 10]
        The VAE features for each residue.
    mask : np.ndarray [n_res]
        The mask of valid residues.
    """
    coords, valid_mask = get_coords_from_pdb(pdb_path, full_backbone=True)
    coords = move_CB(coords, virt_cb=virt_cb) # move CB to virtual position
    vae_features, valid_mask2 = calc_features(coords, valid_mask)

    return vae_features, valid_mask2


def parse_args():
    """Parse arguments for the 3Di script."""
    arg = argparse.ArgumentParser()
    arg.add_argument('encoder', type=str, 
                     help='*.pkl file containing pickled encoder.')
    arg.add_argument('pdb_file', type=str, 
                     help=('Path to PDB file for which to calculate '
                           '3Di sequence.'))
    arg.add_argument('--virt', type=float, nargs=3, default=[270., 0., 2.], 
                     help='Virtual center coordinates in the residue frame.')
    arg.add_argument('--invalid-state', type=str, default='X', 
                     help='Sequence token for residues with invalid state.')
    return arg.parse_args()


def calc_3Di(encoder, pdb_file, virt=[270., 0., 2.]):
    """Calculate the 3Di sequence of a PDB file.
    
    Parameters
    ----------
    encoder : dict
        Dict containing the weights and centroids for the encoder.
    pdb_file : str
        The path to the PDB file for which to calculate the 3Di sequence.
    virt : list, optional
        The virtual C_beta position in the frame of the backbone atoms.

    Returns
    -------
    states : np.ndarray [n_res]
        The 3Di sequence for the PDB file in integer form.
    """
    feat, mask = encoder_features(pdb_file, virt)
    valid_states = discretize(encoder['weights_0'], encoder['bias_0'], 
                              encoder['weights_1'], encoder['bias_1'], 
                              encoder['weights_2'], encoder['bias_2'], 
                              encoder['centroids'], feat[mask])
    states = np.full(len(mask), -1)
    states[mask] = valid_states

    return states


if __name__ == "__main__":
    args = parse_args()
    
    with open(args.encoder, 'rb') as f:
        encoder = pickle.load(f)
    states = calc_3Di(encoder, args.pdb_file, args.virt)

    # 20 letters + X for missing
    LETTERS = 'ACDEFGHIKLMNPQRSTVWYZ'
    seq = ''.join([LETTERS[state] if state != -1 else args.invalid_state 
                   for state in states])
    print(seq)

