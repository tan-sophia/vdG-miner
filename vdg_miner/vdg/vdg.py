import os
import sys
import gzip
import pickle
import numpy as np
import numba as nb
import prody as pr
import xml.etree.ElementTree as ET

from itertools import product
from scipy.spatial.distance import cdist

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalwCommandline
from Bio.Phylo.TreeConstruction import DistanceCalculator

from .constants import *

_dir = os.path.dirname(__file__)
path_to_abple_dict = os.path.join(_dir, '../files/abple_dict.pkl')

with open(path_to_abple_dict, 'rb') as f:
    abple_dict = pickle.load(f)


def get_ABPLE(resname, phi, psi):
    """Get the ABPLE class of a residue given its residue number and angles.
    
    Parameters
    ----------
    resn : str
        The name of the residue.
    phi : float
        The phi angle of the residue (in degrees).
    psi : float
        The psi angle of the residue (in degrees).

    Returns
    -------
    abple : str
        The ABPLE class of the residue.
    """
    try:
        psi = int(np.ceil(psi / 10.0)) * 10
        phi = int(np.ceil(phi / 10.0)) * 10
        if psi == -180:
            psi = -170
        if phi == -180:
            phi = -170
        return abple_dict[resname][psi][phi]
    except ValueError:
        return 'n'
    except KeyError:
        return 'n'


def extract_validation_values(validation_file, chids_resnums):
    """Extract RSCC, RSR, and RSRZ values from a validation XML file.

    Parameters
    ----------
    validation_file : str
        Path to the gzipped validation XML file.
    chids_resnums : list of tuples
        List of tuples of chain IDs and residue numbers for which to extract 
        the RSCC, RSR, and RSRZ values.

    Returns
    -------
    rscc_values : np.ndarray
        List of RSCC values for the specified residues.
    rsr_values : np.ndarray
        List of RSR values for the specified residues.
    rsrz_values : np.ndarray
        List of RSRZ values for the specified residues.
    """
    # Parse the XML file
    with gzip.open(validation_file, 'rt') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Initialize arrays to store the values
    rscc_values = np.zeros(len(chids_resnums))
    rsr_values = np.zeros(len(chids_resnums))
    rsrz_values = np.zeros(len(chids_resnums))

    # rscc_dict = {}
    # rsr_dict = {}
    # rsrz_dict = {}

    # Find the residue elements
    chids, resnums = zip(*chids_resnums)
    for residue in root.findall(".//ModelledSubgroup"):
        chain = residue.get("chain")
        number = int(residue.get("resnum"))
        tup = (chain, number)
        if tup in chids_resnums:
            # Extract RSCC, RSR, and RSRZ values
            try:
                # if tup not in rscc_dict.keys():
                idx = chids_resnums.index(tup)
                rscc_values[idx] = float(residue.get("rscc"))
                rsr_values[idx] = float(residue.get("rsr"))
                rsrz_values[idx] = float(residue.get("rsrz"))
                # rscc_dict[tup] = rscc_values[idx]
                # rsr_dict[tup] = rsr_values[idx]
                # rsrz_dict[tup] = rsrz_values[idx]
                # else:
                #     idx = chids_resnums.index(tup)
                #     rscc_values[idx] = rscc_dict[tup]
                #     rsr_values[idx] = rsr_dict[tup]
                #     rsrz_values[idx] = rsrz_dict[tup]
            except:
                print(('Could not extract validation values ' 
                       'for residue {} in chain {}.').format(number, chain))

    return rscc_values, rsr_values, rsrz_values


# Preprocess pdb_lines and probe_lines into numpy arrays
def preprocess_lines(pdb_lines, probe_lines, atoms_dict={}, do_hash=True):
    """Preprocess PDB and probe lines into numpy arrays for find_neighbors.

    Parameters
    ----------
    pdb_lines : list of str
        List of ATOM lines from a PDB file.
    probe_lines : list of list of str
        List of lines, split by the character ':', from a probe file.
    atoms_dict : dict
        Dictionary of residue names paired with atom names to find in the 
        probe lines.
    do_hash : bool
        Whether or not to hash the output arrays.
    
    Returns
    -------
    pdb_array : np.ndarray [N, ...]
        List of sliced ATOM lines from a PDB file, represented as an array.
    probe_array : np.ndarray [M, ...]
        List of sliced and rearranged lines from a probe file, represented 
        as an array.
    atoms_mask : np.ndarray [M]
        Boolean array indicating whether each probe line contains an atom 
        from the atoms list in the correct residue.
    """
    atoms_mask = np.zeros(len(probe_lines), dtype=np.bool_)
    if do_hash:
        rearrangements_0 = []
        rearrangements_1 = []
        for i, probe_line in enumerate(probe_lines):
            rearrangements_0.append(hash(probe_line[3][11:15] +
                                         probe_line[3][6:11] +
                                         probe_line[3][1:6]))
            rearrangements_1.append(hash(probe_line[4][11:15] +
                                         probe_line[4][6:11] +
                                         probe_line[4][1:6]))
            if not len(atoms_dict):
                atoms_mask[i] = True
            else:
                for res in atoms_dict.keys():   
                    for atom in atoms_dict[res]:
                        if res in probe_line[3] and atom in probe_line[3]:
                            atoms_mask[i] = True
            
        pdb_array = np.array([hash(line[:4] + ' ' + line[5:]) # no altlocs
                              for line in pdb_lines], 
                              dtype=np.int64)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                            dtype=np.int64).T
    else:
        rearrangements_0 = []
        rearrangements_1 = []
        for probe_line in probe_lines:
            rearrangements_0.append(probe_line[3][11:15] +
                                    probe_line[3][6:11] +
                                    probe_line[3][1:6])
            rearrangements_1.append(probe_line[4][11:15] +
                                    probe_line[4][6:11] +
                                    probe_line[4][1:6])
            if not len(atoms_dict):
                atoms_mask[i] = True
            else:
                for res in atoms_dict.keys():   
                    for atom in atoms_dict[res]:
                        if res in probe_line[3] and atom in probe_line[3]:
                            atoms_mask[i] = True

        pdb_array = np.array([line[:4] + ' ' + line[5:] 
                              for line in pdb_lines], 
                             dtype=np.unicode_)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                               dtype=np.unicode_).T

    return pdb_array, probe_array, atoms_mask


@nb.njit
def find_neighbors(pdb_array, probe_array, pdb_coords, probe_coords):
    """Using probe dot positions, find neighboring atoms in a PDB file.
    
    Parameters
    ----------
    pdb_array : np.ndarray [N, ...]
        List of sliced ATOM lines from a PDB file, represented as an array.
    probe_array : np.ndarray [M, ...]
        List of sliced and rearranged lines from a probe file, represented 
        as an array.
    pdb_coords : np.ndarray [N, 3]
        The coordinates of the atoms in the PDB file.
    probe_coords : np.ndarray [M, 3]
        The coordinates of the probe dots.

    Returns
    -------
    neighbors : np.ndarray [N, 2]
        The indices (starting from 0) of neighboring atoms in the PDB file.
    """
    neighbors = -100000 * np.ones((len(probe_coords), 2), dtype=np.int64)
    for i in range(len(probe_array)):
        min_distance_0 = 100000
        min_distance_1 = 100000
        for j in range(len(pdb_array)):
            if probe_array[i, 0] == pdb_array[j]:
                distance = ((pdb_coords[j] - probe_coords[i])**2).sum()
                if distance < min_distance_0:
                    min_distance_0 = distance
                    neighbors[i, 0] = j
            if probe_array[i, 1] == pdb_array[j]:
                distance = ((pdb_coords[j] - probe_coords[i])**2).sum()
                if distance < min_distance_1:
                    min_distance_1 = distance
                    neighbors[i, 1] = j
    return neighbors


'''
@nb.njit
def closest_two_neighbors(coordsA, coordsB, flags):
    """Find the closest two neighbors to a point in coordsB in coordsA.

    Parameters
    ----------
    coordsA : np.ndarray [M, 3]
        The coordinates of the set of points within which to find the 
        closest two neighbors to points in coordsB.
    coordsB : np.ndarray [N, 3] 
        The coordinates of the set of points to which to find the closest 
        two neighbors.
    flags : np.ndarray [M]
        An array of boolean values indicating whether each point in coordsA 
        should go first in any pair of neighbors including it.

    Returns
    -------
    neighbors : np.ndarray [N, 2]
        The indices of the two closest neighbors in coordsA to each point 
        in coordsB.
    """
    neighbors = -100000 * np.ones((len(coordsB), 2), dtype=np.int64)
    for i in range(len(coordsB)):
        flag_found, nonflag_found, two_found = False, False, False
        for j in range(len(coordsA)):
            distance = ((coordsA[j] - coordsB[i])**2).sum()
            if distance <= 4.0:
                if flags[j]: # a flag has been found
                    if not flag_found: # this is the first flag found
                        neighbors[i, 0] = j
                        flag_found = True
                        if nonflag_found: # a nonflag has already been found, 
                                          # ergo two are found and we break
                            two_found = True
                            break
                    else: # this is the second flag found, ergo we break
                        neighbors[i, 1] = j
                        two_found = True
                        break
                else: # a nonflag has been found
                    neighbors[i, 1] = j
                    nonflag_found = True
                    if flag_found: # a flag has also been found, 
                                   # ergo two are found and we break
                        two_found = True
                        break
        if not two_found:
            print(i, coordsB[i])
            raise ValueError('Could not find neighbors.')
    return neighbors
'''


@nb.njit
def percent_identities(alignment):
    """Compute the percent identities of a multiple sequence alignment.

    Parameters
    ----------
    alignment : np.ndarray [N, M]
        The multiple sequence alignment to compute the percent identities of.

    Returns
    -------
    percent : float
        The percent identities of the multiple sequence alignment.
    """
    percent = np.eye(alignment.shape[0])
    for i in range(alignment.shape[0]):
        for j in range(i + 1, alignment.shape[0]):
            mask = np.array([True if alignment[i][k] != b'-' 
                             and alignment[j][k] != b'-' else False
                             for k in range(alignment.shape[1])])
            pct = (alignment[i][mask] == 
                   alignment[j][mask]).sum() / mask.sum()
            percent[i][j] = pct
            percent[j][i] = pct
    return percent


@nb.njit
def greedy(adj):
    """Greedy clustering given an adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray [N x N]
        The adjacency matrix to cluster.
    
    Returns
    -------
    clusters : list
        A list of sets of indices of the clusters.
    """
    if np.any(adj):
        n_neighbors = adj.sum(axis=0)
        max_col = np.argmax(n_neighbors)
        clusters = [list(np.where(adj[max_col])[0])]
        mask = adj[max_col]
        recursive_adj = np.zeros_like(adj)
        recursive_adj[mask][:, mask] = adj[mask][:, mask]
        clusters_next = greedy(recursive_adj)
        if clusters_next is not None:
            clusters += clusters_next
        return clusters
    else:
        return None


class VDG:
    """
    Class to store a vdG, a cluster of local environments of a chemical group.

    More specifically, a vdG (van der Graph) is a collection of local 
    environments consisting of all residues that form contacts (as assessed 
    by probe) with a given chemical group (CG), which itself is a collection 
    of atoms that recurs in protein structures. Each VDG has a fixed number 
    of contacting residues, the identities of which may differ between the 
    distinct local environments that collectively comprise the vdG.
    
    Attributes
    ----------
    cg : str
        SMARTS pattern describing the CG.

    Methods
    -------
    mine_pdb(chain_cluster, rscc=0.8, rsr=0.4, rsrz=2.0, 
             min_seq_sep=7, max_b_factor=60.0, min_occ=0.99)
        Mine all local environments that match the VDG from PDB files.
    remove_redundancy(threshold=0.8)
        Remove sequentially redundant local environments from the vdG.
    cluster(trans_threshold=0.5, rot_threshold=0.5, min_cluster_size=2)
        Cluster the local environments of the vdG.
    merge(other_vdg)
        Merge another vdG object with this vdG object.
    """
    def __init__(self, 
                 cg, 
                 pdb_dir=('/wynton/group/degradolab/nr_pdb/'
                          'clean_final_pdb/'), 
                 validation_dir=('/wynton/group/degradolab/nr_pdb/'
                                 'validation_reports/'), 
                 probe_dir=('/wynton/group/degradolab/nr_pdb/'
                            'probe_files/')):
        self.cg = cg
        if self.cg == 'gn': # TODO: add more CGs
            self.cg_residues = ['ARG']
            self.cg_atoms = {'ARG' : ['NE', 'HE', 'CZ', 
                                      'NH1', 'HH11', 'HH12', 
                                      'NH2', 'HH21', 'HH22']}
            self.cg_hbond_atoms = {'ARG' : ['HE', 'HH11', 'HH12', 
                                            'HH21', 'HH22']}
        if self.cg == 'ccn': # TODO: add more CGs
            self.cg_residues = ['LYS']
            self.cg_atoms = {'LYS' : ['CD', 'CE', 'NZ', 'HZ1/HZ2/HZ3']}
            self.cg_hbond_atoms = {'LYS' : ['HZ1/HZ2/HZ3']}
        if self.cg == 'coo': # TODO: add more CGs
            self.cg_residues = ['ASP', 'GLU']
            self.cg_atoms = {'ASP' : ['CG', 'OD1/OD2'], 
                             'GLU' : ['CD', 'OE1/OE2']}
            self.cg_hbond_atoms = {'ASP' : ['OD1/OD2'], 
                                   'GLU' : ['OE1/OE2']}
        self.contact_types = []
        for res in self.cg_residues:
            # add direct cg-backbone contact types
            for pair in product(self.cg_atoms[res], 
                                ['N', 'H', 'CA', 'HA', 'C', 'O']):
                self.contact_types.append(res + '_' + pair[0] + 
                                          '_' + pair[1])
            # add direct cg-sidechain contact types
            for key, val in protein_atoms.items():
                atoms = []
                for el in val:
                    if type(el) == str and \
                            el not in ['N', 'H', 'CA', 'HA', 'C', 'O']:
                        atoms.append(el)
                    elif type(el) == tuple:
                        atoms.append('/'.join(el))
                for pair in product(self.cg_atoms[res], atoms):
                    self.contact_types.append(res + '_' + pair[0] + '_' + 
                                              pair[1] + '_' + key)
            # add water-mediated cg-backbone contact types
            for pair in product(self.cg_hbond_atoms[res], ['H', 'O']):
                self.contact_types.append(res + '_' + pair[0] + 
                                          '_HOH_' + pair[1])
            # add water-mediated cg-sidechain contact types
            for key, val in protein_hbond_atoms.items():
                atoms = []
                for el in val:
                    if type(el) == str and \
                            el not in ['H', 'O']:
                        atoms.append(el)
                    elif type(el) == tuple:
                        atoms.append('/'.join(el))
                for pair in product(self.cg_hbond_atoms[res], atoms):
                    self.contact_types.append(res + '_' + pair[0] + 
                                              '_HOH_' + pair[1] + '_' + key)            
        self.ABPLE_classes = [''.join(tup) for tup in 
                              product('ABPLE', repeat=3)]
        self.relpos = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'same_chain', 'diff_chain']
        self.fingerprint_cols = \
            self.contact_types + \
            [str(i) + '_' + ac for i, ac in 
             product(range(1, 11), self.ABPLE_classes)] + \
            [str(i) + '_' + str(i + 1) + '_' + rp for i, rp in 
             product(range(1, 10), self.relpos)]
        self.pdb_dir = pdb_dir
        self.validation_dir = validation_dir
        self.probe_dir = probe_dir
        
    def mine_pdb(self, chain_cluster, rscc=0.8, rsr=0.4, rsrz=2.0, 
                 min_seq_sep=7, max_b_factor=60.0, min_occ=0.99):
        """Mine all local environments that match the VDG from PDB files.

        Parameters
        ----------
        chain_cluster : list
            Cluster of chains, homologous at 30% identity or more, from 
            which to mine the VDG. These should be given as strings 
            beginning with a PDB accession code, followed by '_biounit', 
            followed by an integer denoting which biological assembly is 
            under consideration, followed by '_', followed by the chain 
            ID. For example, '1A2P_biounit_1_A' would be a valid chain.
        rscc : float
            Minimum RSCC (real-space correlation coefficient) value for a 
            residue, either containing or contacting the CG, to be mined.
        rsr : float
            Maximum RSR (real-space R-factor) value for a residue, either 
            containing or contacting the CG, to be mined.
        rsrz : float
            Maximum RSRZ (real-space R-factor Z-score) value for a residue, 
            either containing or contacting the CG, to be mined.
        min_seq_sep : int
            Minimum sequence separation between the CG-containing residue 
            and a contacting residue in order for the latter to be mined.
        max_b_factor : float
            Maximum B-factor value for non-hydrogen atoms in a contacting 
            residue to be mined.
        min_occ : float
            Minimum occupancy value for a contacting residue to be mined.
        """
        sc_info = {}
        for mem in chain_cluster:
            # resolve necessary paths
            biounit = mem[:4].upper() + mem[4:-2]
            assert biounit[4:13] == '_biounit_'
            pdb_acc = biounit[:4].lower()
            middle_two = biounit[1:3].lower()
            probe_chain = mem[-1]
            pdb_file = os.path.join(self.pdb_dir, middle_two, biounit + '.pdb')
            probe_file = os.path.join(self.probe_dir, middle_two, 
                                      biounit + '_' + probe_chain + 
                                      '.probe.gz')
            validation_file = os.path.join(self.validation_dir, 
                                           middle_two, pdb_acc, 
                                           pdb_acc + '_validation.xml.gz')
            # read PDB file
            with open(pdb_file, 'r') as f:
                pdb_lines = [line[12:26] for line in f.readlines() 
                            if 'ATOM' in line or 'HETATM' in line]
            pdb = pr.parsePDB(pdb_file)
            pdb_coords = pdb.getCoords()
            res_segnames = np.array([r.getSegname() for r in pdb.iterResidues()])
            res_chids = np.array([r.getChid() for r in pdb.iterResidues()])
            res_resnums = np.array([r.getResnum() for r in pdb.iterResidues()])
            # compute neighbors between pdb atoms and probe dots
            with gzip.open(probe_file, 'rt') as f:
                probe_lines = [line.strip().replace('?', '2').split(':')
                               for line in f.readlines()]
            probe_coords = np.array([[float(line[8]), 
                                      float(line[9]), 
                                      float(line[10])] 
                                     for line in probe_lines])
            contact_types = np.array([line[2] for line in probe_lines])
            # identify neighboring atoms based on probe input
            pdb_array, probe_array, atoms_mask = \
                preprocess_lines(pdb_lines, probe_lines, self.cg_atoms)
            neighbors = find_neighbors(pdb_array, probe_array, 
                                       pdb_coords, probe_coords)
            neighbors_hb_all = neighbors[contact_types == 'hb']
            neighbors_hb = neighbors[np.logical_and(contact_types == 'hb', 
                                                    atoms_mask)]
            neighbors = neighbors[atoms_mask]
            for tup in product(np.unique(pdb.getSegnames()), 
                               np.unique(pdb.getChids())):
                if tup[1] == probe_chain:
                    sc_info[biounit + '_' + '_'.join(tup)] = \
                        {
                            'pdb' : pdb,
                            'validation' : validation_file,
                            'mask' : np.logical_and(
                                pdb.getSegnames() == tup[0], 
                                pdb.getChids() == tup[1]
                            ),  
                            'rmask' : np.logical_and(
                                res_segnames == tup[0], 
                                res_chids == tup[1]
                            ), 
                            'neighbors' : np.empty((0, 2), 
                                                   dtype=np.int64), 
                            'neighbors_hb' : np.empty((0, 2), 
                                                      dtype=np.int64), 
                            'protein_neighbors' : np.empty((0, 2), 
                                                           dtype=np.int64), 
                            'water_bridges' : np.empty((0, 3), 
                                                       dtype=np.int64), 
                            'num_prot_neighbors' : 0, 
                            'num_water_bridges' : 0, 
                            'num_contacts' : 0
                        }
            protein_resindices = \
                np.unique(pdb.select('protein').getResindices())
            water_resindices = np.unique(pdb.select('water').getResindices())
            # determine neighboring protein residues
            resindex_neighbors = pdb.getResindices()[neighbors]
            is_protein_1 = np.isin(resindex_neighbors[:, 0], 
                                   protein_resindices)
            is_protein_2 = np.isin(resindex_neighbors[:, 1], 
                                   protein_resindices)
            prot_neighbors = resindex_neighbors[np.logical_and(is_protein_1, 
                                                               is_protein_2)]
            prot_neighbors = np.unique(prot_neighbors[prot_neighbors[:, 0] != 
                                                      prot_neighbors[:, 1]], 
                                       axis=0)
            # determine water bridges
            water_neighbors = []
            for nbrs in [neighbors_hb, neighbors_hb_all]:
                resindex_neighbors = \
                    pdb.getResindices()[nbrs]
                is_protein_1 = np.isin(resindex_neighbors[:, 0], 
                                    protein_resindices)
                is_protein_2 = np.isin(resindex_neighbors[:, 1], 
                                    protein_resindices)
                is_water_1 = np.isin(resindex_neighbors[:, 0], 
                                    water_resindices)
                is_water_2 = np.isin(resindex_neighbors[:, 1], 
                                    water_resindices)
                water_neighbors.append(
                    np.vstack(
                        [resindex_neighbors[
                                np.logical_and(
                                    is_protein_1, 
                                    is_water_2
                                )
                            ], 
                        resindex_neighbors[
                                np.logical_and(
                                    is_water_1, 
                                    is_protein_2
                                )
                            ][:, ::-1]
                        ]
                    )
                )
            matches = water_neighbors[0][:, 1][:, None] == \
                      water_neighbors[1][:, 1]
            pairs = np.stack(np.where(matches), axis=-1)
            water_bridges = np.hstack((water_neighbors[0][pairs[:, 0]], 
                                       water_neighbors[1][pairs[:, 1], 0:1]))
            water_bridges = np.unique(water_bridges[water_bridges[:, 0] != 
                                                    water_bridges[:, 2]], 
                                      axis=0)
            for key, value in sc_info.items():
                if biounit not in key:
                    continue
                value['neighbors'] = np.vstack(
                    (value['neighbors'], 
                     neighbors[value['mask'][neighbors[:, 0]]])
                )
                value['neighbors_hb'] = np.vstack(
                    (value['neighbors_hb'], 
                     neighbors_hb_all[value['mask'][neighbors_hb_all[:, 0]]])
                )
                value['protein_neighbors'] = np.vstack(
                    (value['protein_neighbors'], 
                     prot_neighbors[value['rmask'][prot_neighbors[:, 0]]])
                )
                value['water_bridges'] = np.vstack(
                    (value['water_bridges'], 
                     water_bridges[value['rmask'][water_bridges[:, 0]]])
                )
                value['num_prot_neighbors'] = len(value['protein_neighbors'])
                value['num_water_bridges'] = len(value['water_bridges'])
                value['num_contacts'] = len(value['protein_neighbors']) + \
                                        len(value['water_bridges'])
        # determine the chain(s) to mine; for each set of symmetry mates 
        # with the same chain ID but different segment IDs, the symmetry 
        # mate with the largest number of contacts (then the lowest segi) 
        # is selected
        clust_contacts = [value['num_contacts'] for value in sc_info.values()]
        ent = list(sc_info.keys())[np.argmax(clust_contacts)]
        biounit = '_'.join(ent.split('_')[:-2])
        middle_two = biounit[1:3].lower()
        pdb = list(sc_info.values())[np.argmax(clust_contacts)]['pdb']
        validation_file = \
            list(sc_info.values())[np.argmax(clust_contacts)]['validation']
        res_segs = np.array([r.getSegname() for r in pdb.iterResidues()])
        res_chids = np.array([r.getChid() for r in pdb.iterResidues()])
        res_resnums = np.array([r.getResnum() for r in pdb.iterResidues()])
        res_resnames = np.array([r.getResname() for r in pdb.iterResidues()])
        res_betas = np.array([r.getBetas()[0] for r in pdb.iterResidues()])
        res_occs = np.array([r.getOccupancies()[0] for r in 
                            pdb.iterResidues()])
        res_phis = 1000. * np.ones_like(res_occs)
        res_psis = 1000. * np.ones_like(res_occs)
        for i, r in enumerate(pdb.iterResidues()):
            try:
                res_phis[i] = pr.calcPhi(r)
                res_psis[i] = pr.calcPsi(r)
            except:
                pass
        prot_neighbors = sc_info[ent]['protein_neighbors']
        water_bridges = sc_info[ent]['water_bridges']
        selstr = 'segname {} and chain {}'.format(ent.split('_')[3], 
                                                  ent.split('_')[4])
        selstr += ' and (resname {}) and name CA'.format(
            ' or resname '.join(self.cg_residues)
        )
        sel = pdb.select(selstr) #TODO: more general CGs
        unique_resindices = np.unique(sel.getResindices())
        fingerprints = []
        environments = []
        for resindex in unique_resindices:
            nbrs = np.concatenate(
                (
                    prot_neighbors[prot_neighbors[:, 0] == resindex][:, 1], 
                    water_bridges[water_bridges[:, 0] == resindex][:, 2]
                )
            )
            _env_idxs = np.concatenate((np.array([resindex]), np.sort(nbrs)))
            chids_resnums = []
            environment = []
            env_idxs = []
            chid0 = res_chids[_env_idxs[0]]
            resnum0 = res_resnums[_env_idxs[0]]
            for i, scr in \
                    enumerate(zip(res_segs[_env_idxs], 
                                  res_chids[_env_idxs], 
                                  res_resnums[_env_idxs])):
                seg, chid, resnum = scr
                d_resnum = np.abs(resnum - resnum0)
                if chid != chid0 or not d_resnum or \
                        d_resnum >= min_seq_sep:
                    if _env_idxs[i] not in env_idxs:
                        chids_resnums.append((chid, resnum))
                        environment.append((biounit, seg, 
                                            chid, resnum))
                        env_idxs.append(_env_idxs[i])
            if len(chids_resnums) < 2:
                continue # No neighbors left.
            env_idxs = np.array(env_idxs)
            rscc_values, rsr_values, rsrz_values = \
                extract_validation_values(validation_file, chids_resnums)
            betas = res_betas[env_idxs]
            occs = res_occs[env_idxs]
            resnames = res_resnames[env_idxs]
            phis = res_phis[env_idxs]
            psis = res_psis[env_idxs]
            resnames_p1 = res_resnames[env_idxs + 1]
            phis_p1 = res_phis[env_idxs + 1]
            psis_p1 = res_psis[env_idxs + 1]
            resnames_m1 = res_resnames[env_idxs - 1]
            phis_m1 = res_phis[env_idxs - 1]
            psis_m1 = res_psis[env_idxs - 1]
            # print('SUMMARY')
            # print('resnums:', res_resnums[env_idxs])
            # print('RSCC: ', rscc_values) # , rscc_values > rscc)
            # print('RSR:  ', rsr_values) # , rsr_values < rsr)
            # print('RSRZ: ', rsrz_values) # , rsrz_values < rsrz)
            # print('Beta: ', betas) # , betas < max_b_factor)
            # print('Occs: ', occs) # , occs > min_occ)
            # print('Phis: ', phis_m1, phis, phis_p1)
            # print('Psis: ', psis_m1, psis, psis_p1)
            if np.all(betas < max_b_factor) and \
                    np.all(occs > min_occ) and \
                    np.all(rscc_values > rscc) and \
                    np.all(rsr_values < rsr) and \
                    np.all(rsrz_values < rsrz) and \
                    np.all(phis != 1000.) and \
                    np.all(phis_p1 != 1000.) and \
                    np.all(phis_m1 != 1000.) and \
                    np.all(psis != 1000.) and \
                    np.all(psis_p1 != 1000.) and \
                    np.all(psis_m1 != 1000.):
                # print(chids_resnums)
                ABPLE = [''.join([get_ABPLE(resname_m1, phi_m1, psi_m1), 
                                  get_ABPLE(resname, phi, psi),
                                  get_ABPLE(resname_p1, phi_p1, psi_p1)])
                         for resname_m1, phi_m1, psi_m1, 
                             resname, phi, psi, 
                             resname_p1, phi_p1, psi_p1 in 
                         zip(resnames_m1[1:], phis_m1[1:], psis_m1[1:], 
                             resnames[1:], phis[1:], psis[1:], 
                             resnames_p1[1:], phis_p1[1:], psis_p1[1:])]
                do_continue = False
                for triplet in ABPLE:
                    if 'n' in triplet:
                        do_continue = True
                if do_continue:
                    # print('n in ABPLE')
                    continue
                # print('All conditions met.')
                fingerprints.append(self.get_fingerprint(env_idxs, 
                                                         sc_info[ent], 
                                                         ABPLE, 
                                                         pdb_file))
                environments.append(environment)
                # print(environment)
                # for tup in environment:
                #     selstr = 'segment {} and chain {} and resnum {}'.format(tup[1], tup[2], tup[3])
                #     assert pdb.select(selstr) is not None and tup[0] in pdb_file
            else:
                pass # print('Some conditions not met.')
        return np.array(fingerprints), environments

    def get_fingerprint(self, env_idxs, ent_sc_info, ABPLE_triplets, pdb_file):
        """Generate a binary fingerprint for a CG environment.

        Parameters
        ----------
        env_idxs : np.ndarray
            The indices of the residues in the environment.
        ent_sc_info : dict
            Dictionary containing information about the chain 
            that has been mined.
        ABPLE_triplets : list
            List of ABPLE classes for each residue in the environment 
            and its neighbors at i - 1 and i + 1.

        Returns
        -------
        fingerprint : np.ndarray
            Binary fingerprint for the CG environment.
        """
        res_chids = np.array([r.getChid() for r in 
                               ent_sc_info['pdb'].iterResidues()])
        res_resnums = np.array([r.getResnum() for r in 
                                ent_sc_info['pdb'].iterResidues()])
        len_fingerprint = len(self.contact_types) + \
                          10 * len(self.ABPLE_classes) + \
                          9 * 11
        fingerprint = np.zeros(len_fingerprint, dtype=np.bool_)
        # set the bits corresponding to the contact types
        for env_idx in env_idxs[1:]:
            is_direct = np.logical_and(
                ent_sc_info['protein_neighbors'][:, 0] == env_idxs[0], 
                ent_sc_info['protein_neighbors'][:, 1] == env_idx, 
            ).sum()
            if is_direct: # direct contact
                # print('IS DIRECT')
                atom_pairs = self.res_contact_to_atom_contacts(
                    env_idxs[0], env_idx, ent_sc_info
                )
                for pair in atom_pairs:
                    cg_resname = ent_sc_info['pdb'].getResnames()[pair[0]]
                    cg_atomname = ent_sc_info['pdb'].getNames()[pair[0]]
                    if cg_atomname not in self.cg_atoms[cg_resname]:
                        # print(cg_atomname, 'not in CG atoms')
                        continue
                    if cg_atomname not in protein_atoms[cg_resname]:
                        for el in protein_atoms[cg_resname]:
                            if cg_atomname in el:
                                cg_atomname = '/'.join(el)
                                break
                    res_resname = ent_sc_info['pdb'].getResnames()[pair[1]]
                    if res_resname not in protein_atoms.keys():
                        continue
                    res_atomname = ent_sc_info['pdb'].getNames()[pair[1]]
                    if res_atomname not in protein_atoms[res_resname]:
                        for el in protein_atoms[res_resname]:
                            if res_atomname in el:
                                res_atomname = '/'.join(el)
                                break
                    if res_atomname in ['N', 'H', 'CA', 'HA', 'C', 'O']:
                        contact_type = '_'.join([cg_resname, 
                                                 cg_atomname,
                                                 res_atomname])
                    else:
                        contact_type = '_'.join([cg_resname, 
                                                 cg_atomname,
                                                 res_atomname, 
                                                 res_resname])
                    # print(contact_type, res_resnums[env_idxs[0]], 
                    #       res_resnums[env_idx], pair[0], pair[1])
                    if contact_type in self.contact_types:
                        fingerprint[self.contact_types.index(contact_type)] = True
                    else:
                        print(contact_type, 'in structure', pdb_file, 
                              'is not in the allowable contact types')
                        raise ValueError()
            bridging_waters = ent_sc_info['water_bridges'][:, 1][
                np.logical_and(
                    ent_sc_info['water_bridges'][:, 0] == 
                        env_idxs[0], 
                    ent_sc_info['water_bridges'][:, 2] == 
                        env_idx,
                )
            ]
            if len(bridging_waters): # water bridge
                # print('WATER BRIDGE')
                for bridging_water in bridging_waters:
                    atom_pairs_0 = self.res_contact_to_atom_contacts(
                        env_idxs[0], bridging_water, ent_sc_info, True
                    )
                    atom_pairs_1 = self.res_contact_to_atom_contacts(
                        env_idx, bridging_water, ent_sc_info, True, True
                    )
                    for pair_0, pair_1 in product(atom_pairs_0, 
                                                  atom_pairs_1):
                        cg_resname = \
                            ent_sc_info['pdb'].getResnames()[pair_0[0]]
                        cg_atomname = \
                            ent_sc_info['pdb'].getNames()[pair_0[0]]
                        if cg_atomname not in self.cg_atoms[cg_resname]:
                            # print(cg_atomname, 'not in CG atoms')
                            continue
                        if cg_atomname not in protein_hbond_atoms[cg_resname]:
                            for el in protein_hbond_atoms[cg_resname]:
                                if cg_atomname in el:
                                    cg_atomname = '/'.join(el)
                                    break
                        res_resname = \
                            ent_sc_info['pdb'].getResnames()[pair_1[0]]
                        res_atomname = \
                            ent_sc_info['pdb'].getNames()[pair_1[0]]
                        if res_atomname not in \
                                protein_hbond_atoms[res_resname]:
                            for el in protein_hbond_atoms[res_resname]:
                                if res_atomname in el:
                                    res_atomname = '/'.join(el)
                                    break
                        if res_atomname in ['N', 'H', 'CA', 'HA', 'C', 'O']:
                            contact_type = '_'.join([cg_resname, 
                                                     cg_atomname,
                                                     'HOH',
                                                     res_atomname])
                        else:
                            contact_type = '_'.join([cg_resname, 
                                                     cg_atomname,
                                                     'HOH',
                                                     res_atomname, 
                                                     res_resname])
                        # print(contact_type, res_resnums[env_idxs[0]], 
                        #       res_resnums[bridging_water], 
                        #       res_resnums[env_idx], pair_0[0], pair_0[1], 
                        #       pair_1[0], pair_1[1])
                        fingerprint[
                            self.contact_types.index(contact_type)
                        ] = True
        # set the bits corresponding to the ABPLE classes
        for i, ABPLE_triplet in enumerate(ABPLE_triplets):
            fingerprint[len(self.contact_types) + 
                        i * len(self.ABPLE_classes) + 
                        self.ABPLE_classes.index(ABPLE_triplet)] = True
        # set the bits corresponding to the relative positions of residues
        for i in range(len(env_idxs) - 2):
            same_chid = (res_chids[env_idxs[i + 2]] == res_chids[env_idxs[i + 1]])
            rel_pos = res_resnums[env_idxs[i + 2]] - res_resnums[env_idxs[i + 1]]
            if same_chid and rel_pos < 10:
                fingerprint[len(self.contact_types) + 
                            10 * len(self.ABPLE_classes) + 
                            i * 11 + rel_pos - 1] = True
            elif same_chid:
                fingerprint[len(self.contact_types) + 
                            10 * len(self.ABPLE_classes) + 
                            i * 11 + 9] = True
            else:
                fingerprint[len(self.contact_types) + 
                            10 * len(self.ABPLE_classes) + 
                            i * 11 + 10] = True
        return fingerprint
            

    @staticmethod
    def res_contact_to_atom_contacts(resindex0, resindex1, ent_sc_info, 
                                     hbond=False, symmetric=False):
        """Return interatomic contacts of a residue-residue contact.

        Parameters
        ----------
        resindex0 : int
            The index of the first residue in the contact.
        resindex1 : int
            The index of the second residue in the contact.
        ent_sc_info : dict
            Dictionary containing information about the chain 
            in which the contact is found.
        hbond : bool
            Whether or not to restrict the contacts to hydrogen bonds.
        symmetric : bool
            Whether or not to treat resindex0 and resindex1 symmetrically.
        """
        mask0 = ent_sc_info['pdb'].getResindices() == resindex0
        mask1 = ent_sc_info['pdb'].getResindices() == resindex1
        if hbond:
            nbr_mask01 = \
                np.logical_and(mask0[ent_sc_info['neighbors_hb'][:, 0]],
                               mask1[ent_sc_info['neighbors_hb'][:, 1]])
            if symmetric:
                nbr_mask10 = \
                    np.logical_and(mask0[ent_sc_info['neighbors_hb'][:, 1]],
                                mask1[ent_sc_info['neighbors_hb'][:, 0]])
                if len(ent_sc_info['neighbors_hb'][nbr_mask01]):
                    return ent_sc_info['neighbors_hb'][nbr_mask01]
                else:
                    return ent_sc_info['neighbors_hb'][nbr_mask10][:, ::-1]
            else:
                return ent_sc_info['neighbors_hb'][nbr_mask01]
        else:
            nbr_mask01 = \
                np.logical_and(mask0[ent_sc_info['neighbors'][:, 0]],
                               mask1[ent_sc_info['neighbors'][:, 1]])
            if symmetric:
                nbr_mask10 = \
                    np.logical_and(mask0[ent_sc_info['neighbors'][:, 1]],
                                   mask1[ent_sc_info['neighbors'][:, 0]])
                if len(ent_sc_info['neighbors'][nbr_mask01]):
                    return ent_sc_info['neighbors'][nbr_mask01]
                else:
                    return ent_sc_info['neighbors'][nbr_mask10][:, ::-1]
            else:
                return ent_sc_info['neighbors'][nbr_mask01]


