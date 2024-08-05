import os
import sys
import gzip
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
# from Bio.Align import MultipleSeqAlignment, PairwiseAligner

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
three_to_one = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D',
                'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G',
                'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', 'LYS' : 'K',
                'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S',
                'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V'}


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
def preprocess_lines(pdb_lines, probe_lines, do_hash=True):
    """Preprocess PDB and probe lines into numpy arrays for find_neighbors.

    Parameters
    ----------
    pdb_lines : list of str
        List of ATOM lines from a PDB file.
    probe_lines : list of list of str
        List of lines, split by the character ':', from a probe file.
    do_hash : bool
        Whether or not to hash the output arrays.
    
    Returns
    -------
    pdb_array : np.ndarray [N, ...]
        List of sliced ATOM lines from a PDB file, represented as an array.
    probe_array : np.ndarray [M, ...]
        List of sliced and rearranged lines from a probe file, represented 
        as an array.
    """
    if do_hash:
        rearrangements_0 = []
        rearrangements_1 = []
        for probe_line in probe_lines:
            rearrangements_0.append(hash(probe_line[3][11:15] +
                                         probe_line[3][6:11] +
                                         probe_line[3][1:6]))
            rearrangements_1.append(hash(probe_line[4][11:15] +
                                         probe_line[4][6:11] +
                                         probe_line[4][1:6]))
            
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
            
        pdb_array = np.array([line[:4] + ' ' + line[5:] 
                              for line in pdb_lines], 
                             dtype=np.unicode_)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                               dtype=np.unicode_).T

    return pdb_array, probe_array


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
    mine_pdb(pdb_file, probe_files, rscc=0.8, rsr=0.4, rsrz=2.0, 
             min_seq_sep=7, max_b_factor=60.0, min_occ=0.99)
        Mine all local environments that match the VDG from a PDB file.
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
        self.pdb_dir = pdb_dir
        self.validation_dir = validation_dir
        self.probe_dir = probe_dir
        
    def mine_pdb(self, biounit, rscc=0.8, rsr=0.4, rsrz=2.0, 
                 min_seq_sep=7, max_b_factor=60.0, min_occ=0.99):
        """Mine all local environments that match the VDG from a PDB file.

        Parameters
        ----------
        biounit : str
            Name of the biounit to mine. Should be a four-letter PDB 
            accession code, followed by '_biounit_', followed by an 
            integer indicating which biological assembly it is.
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
        # resolve necessary pathsvi
        biounit = biounit[:4].upper() + biounit[4:]
        assert biounit[4:13] == '_biounit_'
        pdb_acc = biounit[:4].lower()
        middle_two = biounit[1:3].lower()
        pdb_file = os.path.join(self.pdb_dir, middle_two, biounit + '.pdb')
        probe_files = \
            [os.path.join(self.probe_dir, middle_two, path) for path in 
             os.listdir(os.path.join(self.probe_dir, middle_two)) 
             if biounit in path]
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
        res_betas = np.array([r.getBetas()[0] for r in pdb.iterResidues()])
        res_occs = np.array([r.getOccupancies()[0] for r in 
                             pdb.iterResidues()])
        probe_chains = [probe_file[-10] for probe_file in probe_files]
        sc_info = {'_'.join(tup) :
                   {'mask' : np.logical_and(pdb.getSegnames() == tup[0], 
                                            pdb.getChids() == tup[1]),  
                    'rmask' : np.logical_and(res_segnames == tup[0], 
                                             res_chids == tup[1]), 
                    'neighbors' : np.empty((0, 2), dtype=np.int64), 
                    'neighbors_hb' : np.empty((0, 2), dtype=np.int64), 
                    'protein_neighbors' : np.empty((0, 2), dtype=np.int64), 
                    'water_bridges' : np.empty((0, 3), dtype=np.int64), 
                    'num_prot_neighbors' : 0, 
                    'num_water_bridges' : 0, 
                    'num_contacts' : 0} 
                    for tup in product(np.unique(pdb.getSegnames()), 
                                       np.unique(pdb.getChids()))
                    if tup[1] in probe_chains}
        # read probe files
        for probe_file, probe_chain in zip(probe_files, probe_chains):
            with gzip.open(probe_file, 'rt') as f:
                probe_lines = [line.strip().replace('?', '1').split(':')
                               for line in f.readlines()]
            contact_types = [line[2] for line in probe_lines]
            chid_flags = (pdb.getChids() == probe_chain)
            unique_contact_types = list(set(contact_types))
            contact_type_ids = np.array(
                [unique_contact_types.index(contact_type) 
                 for contact_type in contact_types]
            )
            # compute neighbors between pdb atoms and probe dots
            probe_coords = np.array([[float(line[8]), 
                                      float(line[9]), 
                                      float(line[10])] 
                                     for line in probe_lines])
            # neighbors = closest_two_neighbors(pdb_coords, probe_coords, 
            #                                   chid_flags)
            pdb_array, probe_array = preprocess_lines(pdb_lines, probe_lines)
            neighbors = find_neighbors(pdb_array, probe_array, 
                                       pdb_coords, probe_coords)
            # pdb_array_u, probe_array_u = preprocess_lines(pdb_lines, probe_lines, False)
            # bad_neighbors = np.any(neighbors == -100000, axis=1)
            # print(pdb_file, probe_file)
            # print(probe_array_u[bad_neighbors])
            # sys.exit()
            hb_idx = unique_contact_types.index('hb')
            wh_idx = unique_contact_types.index('wh')
            neighbors_hb = neighbors[contact_type_ids == hb_idx]
            #     neighbors[np.logical_or(contact_type_ids == hb_idx,  
            #                             contact_type_ids == wh_idx)]
            # determine the direct inter-residue contacts
            protein_resindices = \
                np.unique(pdb.select('protein').getResindices())
            water_resindices = np.unique(pdb.select('water').getResindices())
            resindex_neighbors = pdb.getResindices()[neighbors]
            is_protein_1 = np.isin(resindex_neighbors[:, 0], 
                                   protein_resindices)
            is_protein_2 = np.isin(resindex_neighbors[:, 1], 
                                   protein_resindices)
            prot_neighbors = \
                resindex_neighbors[np.logical_and(is_protein_1, 
                                                  is_protein_2)]
            prot_neighbors = np.unique(prot_neighbors[prot_neighbors[:, 0] != 
                                                      prot_neighbors[:, 1]], 
                                       axis=0)
            resindex_neighbors = pdb.getResindices()[neighbors_hb]
            is_protein_1 = np.isin(resindex_neighbors[:, 0], 
                                   protein_resindices)
            is_protein_2 = np.isin(resindex_neighbors[:, 1], 
                                   protein_resindices)
            is_water_1 = np.isin(resindex_neighbors[:, 0], 
                                 water_resindices)
            is_water_2 = np.isin(resindex_neighbors[:, 1], 
                                 water_resindices)
            water_neighbors = np.vstack(
                [resindex_neighbors[np.logical_and(is_protein_1, 
                                                   is_water_2)], 
                 resindex_neighbors[np.logical_and(is_water_1, 
                                                   is_protein_2)][:, ::-1]]
            )
            matches = water_neighbors[:, 1][:, None] == water_neighbors[:, 1]
            pairs = np.stack(np.where(matches), axis=-1)
            water_bridges = np.hstack((water_neighbors[pairs[:, 0]], 
                                       water_neighbors[pairs[:, 1], 0:1]))
            water_bridges = np.unique(water_bridges[water_bridges[:, 0] != 
                                                    water_bridges[:, 2]], 
                                      axis=0)
            for value in sc_info.values():
                value['neighbors'] = np.vstack(
                    (value['neighbors'], 
                     neighbors[value['mask'][neighbors[:, 0]]])
                )
                value['neighbors_hb'] = np.vstack(
                    (value['neighbors_hb'], 
                     neighbors_hb[value['mask'][neighbors_hb[:, 0]]])
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
        sequences = [
                ''.join([three_to_one[r.getResname()] 
                         for i, r in enumerate(pdb.iterResidues())
                         if r.getResname() in three_to_one.keys() and 
                         value['rmask'][i]])
            for value in sc_info.values()
        ]
        keys_list = list(sc_info.keys())
        if len(sequences) > 1:
            msa = pr.buildMSA(sequences, labels=keys_list).getArray()
            for suffix in ['aln', 'dnd', 'fasta']:
                os.remove('Unknown.' + suffix) # clean up after MSA generation
                adj = percent_identities(msa) > 0.3
        else:
            adj = np.ones((1, 1), dtype=np.bool_)
        unique_ents = []
        for cluster in greedy(adj):
            c_keys = np.array([keys_list[i] for i in cluster])
            clust_contacts = [sc_info[key]['num_contacts'] for key in c_keys]
            ent = c_keys[np.argmax(clust_contacts)]
            prot_neighbors = sc_info[ent]['protein_neighbors']
            water_bridges = sc_info[ent]['water_bridges']
            selstr = 'segname {} and chain {}'.format(ent.split('_')[0], 
                                                      ent.split('_')[1])
            print(ent, selstr)
            sel = pdb.select(selstr + ' and resname ARG and name CA') #TODO: more general CGs
            unique_resindices = sel.getResindices()
            for resindex in unique_resindices:
                _env_idxs = np.concatenate(
                    (np.array([resindex]), 
                     prot_neighbors[prot_neighbors[:, 0] == resindex][:, 1], 
                     water_bridges[water_bridges[:, 0] == resindex][:, 2])
                )
                chids_resnums = []
                env_idxs = []
                chid0 = res_chids[_env_idxs[0]]
                resnum0 = res_resnums[_env_idxs[0]]
                for i, chid_resnum in \
                        enumerate(zip(res_chids[_env_idxs], 
                                      res_resnums[_env_idxs])):
                    chid, resnum = chid_resnum
                    d_resnum = np.abs(resnum - resnum0)
                    if chid != chid0 or not d_resnum or \
                            d_resnum >= min_seq_sep:
                        if _env_idxs[i] not in env_idxs:
                            chids_resnums.append((chid, resnum))
                            env_idxs.append(_env_idxs[i])
                env_idxs = np.array(env_idxs)
                if len(chids_resnums) < 2:
                    continue # No neighbors left.
                rscc_values, rsr_values, rsrz_values = \
                    extract_validation_values(validation_file, chids_resnums)
                betas = res_betas[env_idxs]
                occs = res_occs[env_idxs]
                print('SUMMARY')
                print('resnums:', res_resnums[env_idxs])
                print('RSCC: ', rscc_values, rscc_values > rscc)
                print('RSR:  ', rsr_values, rsr_values < rsr)
                print('RSRZ: ', rsrz_values, rsrz_values < rsrz)
                print('Beta: ', betas, betas < max_b_factor)
                print('Occs: ', occs, occs > min_occ)
                if np.all(betas < max_b_factor) and \
                        np.all(occs > min_occ) and \
                        np.all(rscc_values > rscc) and \
                        np.all(rsr_values < rsr) and \
                        np.all(rsrz_values < rsrz):
                    print(chids_resnums)
                    print('All conditions met.')