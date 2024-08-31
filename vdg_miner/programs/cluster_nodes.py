import os
import sys
import argparse
from itertools import permutations
from collections import defaultdict

import numpy as np
import prody as pr

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
       'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
       'TYR', 'VAL']

def permute_with_symmetry(symmetry_classes):
    """Get all possible permutation arrays that preserve symmetry classes.
    
    Parameters
    ----------
    symmetry_classes : list
        List of integers representing the symmetry classes of the elements.
        
    Returns
    -------
    valid_permutations : list
        List of valid permutations that preserve symmetry classes.
    """
    elements = list(range(len(symmetry_classes)))
    # Get all possible permutations
    all_permutations = permutations(elements)
    # Filter permutations based on symmetry classes
    valid_permutations = []
    for perm in all_permutations:
        is_valid = True
        for i, p in enumerate(perm):
            # Compare the symmetry classes of elements in the original list
            # and the permuted list at the same positions
            if symmetry_classes[elements.index(p)] != symmetry_classes[i]:
                is_valid = False
                break
        if is_valid:
            valid_permutations.append(np.array(perm))
    return valid_permutations

def greedy(adj_mat, min_cluster_size=2):
    """Greedy clustering algorithm based on an adjacency matrix.
        
        Takes an adjacency matrix as input.
        All values of adj_mat are 1 or 0:  1 if <= to cutoff, 
        0 if > cutoff.

        The diagonal of adj_mat should be 0.

    Parameters
    ----------
    adj_mat : scipy.sparse.csr_matrix
        Adjacency matrix of the graph.
    min_cluster_size : int, optional
        Minimum size of a cluster, by default 2.

    Returns
    -------
    all_mems : list
        List of arrays of cluster members.
    cents : list
        List of cluster centers.
    """
    if not isinstance(adj_mat, csr_matrix):
        try:
            adj_mat = csr_matrix(adj_mat)
        except:
            print('adj_mat distance matrix must be scipy csr_matrix '
                  '(or able to convert to one)')
            return

    assert adj_mat.shape[0] == adj_mat.shape[1], \
        'Distance matrix is not square.'

    all_mems = []
    cents = []
    indices = np.arange(adj_mat.shape[0])

    try:
        while adj_mat.shape[0] > 0:

            cent = adj_mat.sum(axis=1).argmax()
            row = adj_mat.getrow(cent)
            tf = ~row.toarray().astype(bool)[0]
            mems = indices[~tf]

            if len(mems) < min_cluster_size:
                [cents.append(i) for i in indices]
                [all_mems.append(np.array([i])) for i in indices]
                break

            cents.append(indices[cent])
            all_mems.append(mems)

            indices = indices[tf]
            adj_mat = adj_mat[tf][:, tf]
    except KeyboardInterrupt:
        pass

    return all_mems, cents


def kabsch(X, Y):
    """Rotate and translate X into Y to minimize the SSD between the two, 
       and find the derivatives of the SSD with respect to the entries of Y. 
       
       Implements the SVD method by Kabsch et al. (Acta Crystallogr. 1976, 
       A32, 922).

    Parameters
    ----------
    X : np.array [M x N x 3]
        Array of M sets of mobile coordinates (N x 3) to be transformed by a 
        proper rotation to minimize sum squared displacement (SSD) from Y.
    Y : np.array [M x N x 3]
        Array of M sets of stationary coordinates relative to which to 
        transform X.

    Returns
    -------
    R : np.array [M x 3 x 3]
        Proper rotation matrices required to transform each set of coordinates 
        in X such that its SSD with the corresponding coordinates in Y is 
        minimized.
    t : np.array [M x 3]
        Translation matrix required to transform X such that its SSD with Y 
        is minimized.
    ssd : np.array [M]
        Sum squared displacement after alignment for each pair of coordinates.
    """
    # compute R using the Kabsch algorithm
    Xbar, Ybar = np.mean(X, axis=1, keepdims=True), \
                 np.mean(Y, axis=1, keepdims=True)
    Xc, Yc = X - Xbar, Y - Ybar
    H = np.matmul(np.transpose(Xc, (0, 2, 1)), Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.matmul(U, Vt)))
    D = np.zeros((X.shape[0], 3, 3))
    D[:, 0, 0] = 1.
    D[:, 1, 1] = 1.
    D[:, 2, 2] = d
    R = np.matmul(U, np.matmul(D, Vt))
    t = (Ybar - np.matmul(Xbar, R)).reshape((-1, 3))
    # compute SSD from aligned coordinates XR
    XRmY = np.matmul(Xc, R) - Yc
    ssd = np.sum(XRmY ** 2, axis=(1, 2))
    return R, t, ssd


def cluster_structures(directory, cg='gn', cutoff=1.0, 
                       idxs=None, symmetry_classes=None):
    """Greedily cluster the structures in a directory based on RMSD.

    Parameters
    ----------
    directory : str
        The directory containing the structures to cluster.\
    cg : str, optional
        The chemical group at the center of each structure, by default 'gn'.
    cutoff : float, optional
        The RMSD cutoff for greedy clustering, by default 1.0.
    idxs : list, optional
        Indices of CG atoms on which to cluster, by default None. If None, 
        all CG atoms are used.
    symmetry_classes : list, optional
        Integers representing the symmetry classes of the CG atoms on which
        clustering is to be performed. If provided, should have the same
        length as idxs. If not provided, the atoms are assumed to be
        symmetrically inequivalent.
    """
    subdirs = directory.split('/')
    node_aas = []
    node_aa_counts = {aa : 0 for aa in aas}
    for i, subdir in enumerate(subdirs):
        flag = False
        for aa in aas:
            if subdir[:3] == aa:
                node_aas.append((aa, node_aa_counts[aa]))
                node_aa_counts[aa] += 1
                flag = True
                break
        if i == len(subdirs) - 1 and not flag:
            return
    print(directory)
    pdbs = [f for f in os.listdir(directory) if f.endswith('.pdb')]
    structs = [pr.parsePDB(os.path.join(directory, pdb)) for pdb in pdbs]
    if idxs is None:
        occs0 = structs[0].getOccupancies()
        idxs = list(range((occs0 >= 3.).sum()))
    if symmetry_classes is None:
        symmetry_classes = list(range(len(idxs)))
    assert len(symmetry_classes) == len(idxs), \
        'Length of symmetry_classes must match length of idxs.'
    perms = permute_with_symmetry(symmetry_classes)
    coords = []
    for i in range(len(pdbs)):
        occs = structs[i].getOccupancies()
        names = structs[i].getNames()
        resnames = structs[i].getResnames()
        all_coords = structs[i].getCoords()
        cg_idxs = np.array(
            [np.argwhere(occs == 3. + 0.1 * idx)[0][0] for idx in idxs]
        )
        coords_to_add = []
        for perm in perms:
            perm_coords = []
            perm_coords.append(
                all_coords[cg_idxs[perm]]
            )
            for j in range(len(node_aas)):
                for name in ['N', 'CA', 'C']: #, O]:
                    mask = np.logical_and.reduce((occs > 1.,
                                                  names == name,
                                                  resnames == node_aas[j][0]))
                    perm_coords.append(
                        all_coords[mask][node_aas[j][1]:node_aas[j][1]+1]
                    )
            coords_to_add.append(np.vstack(perm_coords))
        coords += coords_to_add
    coords = np.array(coords).reshape((len(pdbs), len(perms), -1, 3))
    coords = coords.transpose((1, 0, 2, 3))
    M = coords.shape[1]
    N = len(idxs) + \
        len(node_aas) * 3 # number of atoms in CG plus N,CA,C for each aa
    # find minimal-RMSD alignments between all pairs of structures
    triu_indices = np.triu_indices(M, 1)
    L = len(triu_indices[0])
    R, t, ssd = np.zeros((L, 3, 3)), np.zeros((L, 3)), 10000. * np.ones(L)
    for i in range(coords.shape[0]):
        _R, _t, _ssd = kabsch(coords[0][triu_indices[0]], 
                              coords[i][triu_indices[1]])
        R[_ssd < ssd] = _R[_ssd < ssd]
        t[_ssd < ssd] = _t[_ssd < ssd]
        ssd[_ssd < ssd] = _ssd[_ssd < ssd]
    A = np.eye(M, dtype=int)
    A[triu_indices] = (ssd <= N * cutoff ** 2).astype(int)
    A = A + A.T
    all_mems, cents = greedy(A)
    if set([len(cluster) for cluster in all_mems]) == {1}:
        return
    cluster_dirname = os.path.join(directory, 'clusters')
    os.makedirs(cluster_dirname, exist_ok=True)
    cluster_num = 1
    for cluster, cent in zip(all_mems, cents):
        assert cent in cluster, f'Centroid {cent} not in cluster {cluster}.'
        if len(cluster) == 1:
            continue
        cluster_subdirname = \
            os.path.join(cluster_dirname, 
                         'cluster_{}_size_{}'.format(cluster_num, 
                                                     len(cluster)))
        os.makedirs(cluster_subdirname, exist_ok=True)
        for i, el in enumerate(cluster):
            pdb_name = pdbs[el]
            cl_struct = pr.parsePDB(os.path.join(directory, pdb_name))
            if el == cent:
                pdb_name = pdb_name[:-4] + '_centroid.pdb'
            else:
                i_mobile = np.logical_and(triu_indices[0] == el,
                                          triu_indices[1] == cent)
                cent_mobile = np.logical_and(triu_indices[0] == cent,
                                             triu_indices[1] == el)
                if np.any(i_mobile):
                    idx = np.argwhere(i_mobile)[0][0]
                    _R = R[idx]
                    _t = t[idx]
                    rmsd = np.sqrt(ssd[idx] / N)
                else:
                    idx = np.argwhere(cent_mobile)[0][0]
                    _R = R[idx].T
                    _t = -np.dot(_R, t[idx])
                    rmsd = np.sqrt(ssd[idx] / N)
                try:
                    cl_struct.setCoords(
                        np.dot(cl_struct.getCoords(), _R) + _t
                    )
                    if N > len(idxs) + 3:
                        print('RMSD =', rmsd)
                except:
                    print(cl_struct.getCoords().shape, 
                          cent, el, _R.shape, _t.shape)
                    sys.exit()
            pr.writePDB(os.path.join(cluster_subdirname, pdb_name), 
                        cl_struct)
        cluster_num += 1


def cluster_structures_at_depth(starting_dir, target_depth, cutoff=1.0, 
                                idxs=None, symmetry_classes=None):
    """
    Traverses the directory tree starting from `starting_dir` and clusters 
    the structures in each sub-tree at a specified depth.

    Parameters
    ----------
    starting_dir : str
        The root directory from which to start the traversal.
    target_depth : int
        The depth below which directories should be clustered.
    cutoff : float, optional
        The RMSD cutoff for greedy clustering, by default 1.0.
    idxs : list, optional
        Indices of CG atoms on which to cluster, by default None. If None, 
        all CG atoms are used.
    symmetry_classes : list, optional
        Integers representing the symmetry classes of the CG atoms on which
        clustering is to be performed. If provided, should have the same
        length as idxs. If not provided, the atoms are assumed to be
        symmetrically inequivalent.
    """
    cg = starting_dir.split('/')[-1].split('_')[0]
    def get_depth(path):
        return path[len(starting_dir):].count(os.sep)

    for root, dirs, files in os.walk(starting_dir, topdown=False):
        current_depth = get_depth(root)

        if current_depth >= target_depth and 'clusters' not in root and \
                root.split('_')[-1] != '1':
            # Cluster the structures in the current directory
            cluster_structures(root, cg=cg, cutoff=cutoff, idxs=idxs, 
                               symmetry_classes=symmetry_classes)


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster structures in a '
                                     'directory tree based on RMSD.')
    parser.add_argument('starting_dir', type=str, help='The root directory '
                        'from which to start the traversal.')
    parser.add_argument('target_depth', type=int, help='The depth below which '
                        'directories should be clustered.')
    parser.add_argument('cutoff', type=float, help='The RMSD cutoff for '
                        'greedy clustering.')
    parser.add_argument('-i', '--idxs', nargs='+', type=int, 
                        help='Indices of CG atoms on which to cluster.')
    parser.add_argument('-s', '--symmetry-classes', nargs='+', type=int,
                        help='Integers representing the symmetry classes of '
                             'the CG atoms on which clustering is to be '
                             'performed. If provided, should have the same '
                             'length as idxs. If not provided, the atoms '
                             'are assumed to be symmetrically inequivalent.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cluster_structures_at_depth(args.starting_dir, args.target_depth,
                                args.cutoff, args.idxs, args.symmetry_classes)