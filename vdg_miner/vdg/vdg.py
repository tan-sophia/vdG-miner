import os
import gzip
import numpy as np
import numba as nb
import prody as pr

from itertools import product
from scipy.spatial.distance import cdist

from vdg_miner.constants import *
from vdg_miner.database.readxml import extract_residue_validation_values


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
        Dictionary of residue names as keys paired with lists of atom names 
        as values to find in the probe lines. Default: {}.
    do_hash : bool
        Whether or not to hash the output arrays. Default: True.
    
    Returns
    -------
    pdb_array : np.ndarray [N, ...]
        List of sliced ATOM lines from a PDB file, represented as an array.
    probe_array : np.ndarray [M, ...]
        List of sliced and rearranged lines from a probe file, represented 
        as an array.
    atoms_mask : np.ndarray [M]
        Boolean array indicating whether each probe line has as its first 
        atom an atom in the atoms_dict.
    water_mask : np.ndarray [M]
        Boolean array indicating whether each probe line has as its first
        atom an atom in a water molecule.
    """
    atoms_mask = np.zeros(len(probe_lines), dtype=np.bool_)
    water_mask = np.zeros(len(probe_lines), dtype=np.bool_)
    if do_hash: # hash the arrays for speed
        # rearrange the sections of the probe lines that contain atom info 
        # to match PDB format for both the first and second atoms
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
                water_mask[i] = True
            else:
                for res in atoms_dict.keys():
                    for cg_atom_list in atoms_dict[res]:
                        for atom in cg_atom_list:
                            if res in probe_line[3] and atom in probe_line[3]:
                                atoms_mask[i] = True
                if 'HOH' in probe_line[3] and 'HOH' not in probe_line[4]:
                    water_mask[i] = True
        # 12:26 is atom name, resname, chain, and resnum
        pdb_array = np.array([hash(line[12:26]) for line in pdb_lines], 
                              dtype=np.int64)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                               dtype=np.int64).T
    else:
        # rearrange the sections of the probe lines that contain atom info 
        # to match PDB format for both the first and second atoms
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
                water_mask[i] = True
            else:
                for res in atoms_dict.keys():   
                    for atom in atoms_dict[res]:
                        if res in probe_line[3] and atom in probe_line[3]:
                            atoms_mask[i] = True
                if 'HOH' in probe_line[3] and 'HOH' not in probe_line[4]:
                    water_mask[i] = True
        # 12:26 is atom name, resname, chain, and resnum
        pdb_array = np.array([line[12:26] for line in pdb_lines], 
                             dtype=np.unicode_)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                               dtype=np.unicode_).T

    return pdb_array, probe_array, atoms_mask, water_mask


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
    mine_pdb(chain_cluster, cg_df=None, rscc=0.8, rsr=0.4, rsrz=2.0, 
             min_seq_sep=7, max_b_factor=60.0, min_occ=0.99)
        Mine all local environments that match the VDG from PDB files.
    remove_redundancy(threshold=0.8)
        Remove sequentially redundant local environments from the vdG.
    cluster(trans_threshold=0.5, rot_threshold=0.5, min_cluster_size=2)
        Cluster the local environments of the vdG.
    merge(other_vdg)
        Merge another vdG object with this vdG object.
    """
    def __init__(self, cg, pdb_dir, probe_dir, validation_dir, 
                 cg_natoms=None):
        if cg in cg_resnames.keys(): # CG is proteinaceous
            self.cg_resnames = cg_resnames[cg]
            self.cg_atoms = \
                {res : cg_atoms[cg][res] for res in cg_resnames[cg]}
        else:
            assert cg_natoms is not None
            self.cg_resnames = ['XXX']
            self.cg_atoms = \
                {'XXX' : ['atom' + str(i) 
                          for i in range(cg_natoms)]}
        self.max_nbrs = 15
        self.contact_cols = []
        self.ABPLE_cols = \
            [str(i) + '_' + ac for i, ac in
             product(range(1, self.max_nbrs + 1), ABPLE_triplets)]
        self.relpos_cols = \
            [str(i) + '_' + str(i + 1) + '_' + rp for i, rp in
                product(range(1, self.max_nbrs), relpos)]        
        self.pdb_dir = pdb_dir
        self.probe_dir = probe_dir
        self.validation_dir = validation_dir
        # convenience attributes to prevent unnecessary file reads
        self.prev_pdb_file = ''
        self.prev_pdb = None
        self.prev_pdb_lines = []

    def update_sc_info(self, sc_info, segi, chain, 
                       pdb_file, probe_file, cg_match_dict=None):
        """Update the sc_info dict with information from a structure.
        
        Parameters
        ----------
        sc_info : dict
            Dictionary containing information about the segment/chain pairs 
            that have been mined.
        pdb_file : str
            Path to the PDB file corresponding to the structure.
        probe_file : str
            Path to the probe.gz file encoding the contacts in the structure.
        segi : str
            Segment ID to mine from the structure.
        chain : str
            Chain ID to mine from the structure.
        cg_match_dict : dict, optional
            Dictionary of matching CGs in ligands with keys as tuples of 
            (struct_name, seg, chain, resnum, resname) for the ligand and 
            values as lists containing the list of atom names for each match 
            to the CG. Used for non-protein CGs. Default: None.
        """
        # read PDB file
        if self.prev_pdb_file == pdb_file:
            pdb = self.prev_pdb
            pdb_lines = self.prev_pdb_lines
        else:
            pdb_lines = []
            with open(pdb_file, 'rb') as f:
                for b_line in f:
                    if b_line.startswith(b'ATOM') or \
                            b_line.startswith(b'HETATM'):
                        pdb_lines.append(b_line.decode('utf-8'))
            pdb = pr.parsePDB(pdb_file)
            self.prev_pdb_file = pdb_file
            self.prev_pdb = pdb
            self.prev_pdb_lines = pdb_lines
        pdb_coords = pdb.getCoords()
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
        struct_name = pdb_file.split('/')[-1].split('.')[0]
        if 'XXX' in self.cg_atoms.keys(): # non-proteinaceous CG
            cg_atoms_dict = {key[4] : val 
                             for key, val in cg_match_dict.items() 
                             if key[0] == struct_name}
        else: # proteinaceous CG
            cg_atoms_dict = {key : [val] 
                             for key, val in self.cg_atoms.items()}
        # preprocess pdb and probe lines as integers for fast matching
        pdb_array, probe_array, atoms_mask, water_mask = \
            preprocess_lines(pdb_lines, probe_lines, cg_atoms_dict)
        # find matches of pdb and probe lines with numba to determine which 
        # atoms are neighbors (necessary because probe does not output segis)
        neighbors = \
            find_neighbors(pdb_array, probe_array, pdb_coords, probe_coords)
        if -100000 in neighbors:
            return
        neighbors_hb = \
            neighbors[np.logical_and(contact_types == 'hb', atoms_mask)]
        neighbors_hb_wat = \
            neighbors[np.logical_and(contact_types == 'hb', water_mask)]
        neighbors = neighbors[atoms_mask]
        resindex_neighbors = pdb.getResindices()[neighbors]
        resindex_neighbors_wat = pdb.getResindices()[neighbors_hb_wat]
        # account for index of CG in residue for each neighbor
        neighbor_resnames = pdb.getResnames()[neighbors[:, 0]]
        neighbor_atomnames = pdb.getNames()[neighbors[:, 0]]
        cg_idxs = np.zeros((len(neighbors), 1), dtype=np.int64)
        for i, (rn, an) in enumerate(zip(neighbor_resnames, 
                                         neighbor_atomnames)):
            if rn in cg_atoms_dict.keys():
                for j, an_list in enumerate(cg_atoms_dict[rn]):
                    if an in an_list:
                        cg_idxs[i] = j + 1
        resindex_neighbors = np.hstack([cg_idxs, resindex_neighbors])
        # determine neighboring non-water residues
        nonwater_resindices = \
            np.unique(pdb.select('not water').getResindices())
        nonwater_neighbors = \
            resindex_neighbors[np.isin(resindex_neighbors[:, 2], 
                                       nonwater_resindices)]
        nonwater_neighbors = \
            np.unique(nonwater_neighbors[nonwater_neighbors[:, 1] != 
                                         nonwater_neighbors[:, 2]], 
                      axis=0) # remove self-contacts
        # determine water bridges
        water_sel = pdb.select('water')
        if water_sel is not None:
            water_resindices = np.unique(water_sel.getResindices())
            nonwater_water_neighbors = \
                resindex_neighbors[np.isin(resindex_neighbors[:, 2], 
                                           water_resindices)]
            water_nonwater_neighbors = \
                resindex_neighbors_wat[np.isin(resindex_neighbors_wat[:, 1], 
                                               nonwater_resindices)]
            unique_water_neighbors = \
                np.unique(np.vstack([nonwater_water_neighbors[:, 2], 
                                     water_nonwater_neighbors[:, 0]]))
            water_bridges = []
            for k in unique_water_neighbors:
                nonwater_0 = \
                    nonwater_water_neighbors[
                        nonwater_water_neighbors[:, 2] == k
                    ][:, :2]
                nonwater_1 = \
                    water_nonwater_neighbors[
                        water_nonwater_neighbors[:, 0] == k
                    ][:, 1]
                for i, j in np.unique(nonwater_0, axis=0):
                    for l in np.unique(nonwater_1):
                        if j != l and [i, j, k, l] not in water_bridges:
                            water_bridges.append([i, j, k, l])
            water_bridges = np.array(water_bridges)
        else:
            water_bridges = np.empty((0, 4), dtype=np.int64)
        # create dictionary entry for segment/chain pair of interest
        res_segnames = np.array([r.getSegname() for r in pdb.iterResidues()])
        res_chids = np.array([r.getChid() for r in pdb.iterResidues()])
        mask = np.logical_and(pdb.getSegnames() == segi, 
                              pdb.getChids() == chain)
        rmask = np.logical_and(res_segnames == segi, 
                               res_chids == chain)
        neighbors_masked = \
            neighbors[mask[neighbors[:, 0]]]
        neighbors_hb_masked = \
            neighbors_hb[mask[neighbors_hb[:, 0]]]
        nonwater_neighbors_masked = \
            nonwater_neighbors[rmask[nonwater_neighbors[:, 1]]]
        water_bridges_masked = \
            water_bridges[rmask[water_bridges[:, 1]]]
        sc_info['_'.join([struct_name, segi, chain])] = \
            {
                'pdb' : pdb,
                'cg_atoms_dict' : cg_atoms_dict,
                'mask' : mask,
                'rmask' : rmask,
                'neighbors' : neighbors_masked, 
                # 'neighbors_hb' : neighbors_hb_masked, 
                'nonwater_neighbors' : nonwater_neighbors_masked,
                'water_bridges' : water_bridges_masked, 
                'num_contacts' : len(nonwater_neighbors_masked) + 
                                 len(water_bridges_masked)
            }
        
    def mine_pdb(self, chain_cluster=None, cg_match_dict=None, 
                 rscc=0.8, rsr=0.4, rsrz=2.0, min_seq_sep=7, 
                 max_b_factor=60.0, min_occ=0.99):
        """Mine all local environments that match the VDG from PDB files.

        Parameters
        ----------
        chain_cluster : list, optional
            Cluster of chains, homologous at 30% identity or more, from which 
            to mine the VDG. These should be given as strings beginning with 
            a PDB accession code, followed by '_biounit_', followed by an 
            integer denoting which biological assembly is under consideration, 
            followed by '_', followed by the chain ID. For example, 
            '1A2P_biounit_1_A' would be a valid chain. If the final '_' and 
            chain are omitted, all chains in the biounit are mined. Default: 
            None, but either chain_cluster or cg_match_dict must not be None.
        cg_match_dict : dict, optional
            Dictionary of matching CGs in ligands with keys as tuples of 
            (struct_name, seg, chain, resnum, resname) for the ligand and 
            values as lists containing the list of atom names for each match 
            to the CG. Used for non-protein CGs. Default: None, but either 
            chain_cluster or cg_match_dict must not be None.
        rscc : float, optional
            Minimum RSCC (real-space correlation coefficient) value for a 
            residue, either containing or contacting the CG, to be mined.
        rsr : float, optional
            Maximum RSR (real-space R-factor) value for a residue, either 
            containing or contacting the CG, to be mined.
        rsrz : float, optional
            Maximum RSRZ (real-space R-factor Z-score) value for a residue, 
            either containing or contacting the CG, to be mined.
        min_seq_sep : int, optional
            Minimum sequence separation between the CG-containing residue 
            and a contacting residue in order for the latter to be mined.
        max_b_factor : float, optional
            Maximum B-factor value for non-hydrogen atoms in a contacting 
            residue to be mined.
        min_occ : float, optional
            Minimum occupancy value for a contacting residue to be mined.

        Returns
        -------
        fingerprint_labels : list
            List of lists of labels at which the fingerprints for each 
            vdG should be True.
        environments : list
            List of local environments for the VDG.
        """
        sc_info = {} # dictionary of information on segment/chain pairs
        if chain_cluster is not None:
            for mem in chain_cluster:
                # resolve necessary paths
                biounit = '_'.join(mem.split('_')[:-2])
                assert biounit[4:13] == '_biounit_'
                segi, chain = mem.split('_')[-2:]
                pdb_acc = biounit[:4].lower()
                middle_two = biounit[1:3].lower()
                pdb_file = \
                    os.path.join(self.pdb_dir, middle_two, biounit + '.pdb')
                probe_file = \
                    os.path.join(self.probe_dir, middle_two, 
                                 '_'.join(biounit, chain + '.probe.gz'))
                #                '_'.join(biounit, segi, chain, '.probe.gz'))
                validation_file = \
                    os.path.join(self.validation_dir, middle_two, pdb_acc, 
                                 pdb_acc + '_validation.xml.gz')
                self.update_sc_info(sc_info, segi, chain, 
                                    pdb_file, probe_file, 
                                    cg_match_dict)
        elif cg_match_dict is not None:
            for key in set([key[:3] for key in cg_match_dict.keys()]):
                struct_name, segi, chain = key
                middle_two = struct_name[1:3].lower()
                pdb_file = os.path.join(self.pdb_dir, middle_two, 
                                        struct_name + '.pdb')
                probe_file = os.path.join(self.probe_dir, middle_two, 
                                          struct_name + '.probe.gz')
                # TODO: change for long-term database file names with 
                #       segi and chain
                validation_file = \
                    os.path.join(self.validation_dir, middle_two, 
                                 struct_name + '_validation.xml.gz')
                self.update_sc_info(sc_info, segi, chain, 
                                    pdb_file, probe_file, 
                                    cg_match_dict)
        else:
            raise ValueError('Either chain_cluster or cg_match_dict '
                             'must not be None.')
        # determine the chain(s) to mine; for each set of symmetry mates 
        # with the same chain ID but different segment IDs, the symmetry 
        # mate with the largest number of contacts (then the lowest segi) 
        # is selected
        clust_contacts = [value['num_contacts'] for value in sc_info.values()]
        if not len(clust_contacts):
            return [], []
        ent = list(sc_info.keys())[np.argmax(clust_contacts)]
        biounit = '_'.join(ent.split('_')[:-2])
        middle_two = biounit[1:3].lower()
        pdb = sc_info[ent]['pdb']
        validation_file = os.path.join(self.validation_dir, 
                                       middle_two, 
                                       biounit[:4].lower(), 
                                       biounit[:4].lower() + 
                                       '_validation.xml.gz')
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
        if '__' in ent: # no segment name
            selstr = 'chain {}'.format(ent.split('_')[-1])
        else:
            selstr = 'segname {} and chain {}'.format(ent.split('_')[-2], 
                                                      ent.split('_')[-1])
        selstr += ' and (resname {})'.format(
            ' or resname '.join(sc_info[ent]['cg_atoms_dict'].keys())
        )
        sel = pdb.select(selstr)
        nonwater_neighbors = sc_info[ent]['nonwater_neighbors']
        water_bridges = sc_info[ent]['water_bridges']
        unique_cg_idxs = np.unique(np.hstack([nonwater_neighbors[:, 0], 
                                              water_bridges[:, 0]]))
        unique_resindices = np.unique(sel.getResindices())
        water_bridges = sc_info[ent]['water_bridges']
        fingerprint_labels = []
        environments = []
        for cg_idx in unique_cg_idxs[unique_cg_idxs > 0]:
            for resindex in unique_resindices:
                nw_mask = np.logical_and(
                    nonwater_neighbors[:, 0] == cg_idx, 
                    nonwater_neighbors[:, 1] == resindex
                )
                wb_mask = np.logical_and(
                    water_bridges[:, 0] == cg_idx, 
                    water_bridges[:, 1] == resindex
                )
                nbrs = np.concatenate((nonwater_neighbors[nw_mask][:, 2], 
                                       water_bridges[wb_mask][:, 3]))
                _env_idxs = np.concatenate((np.array([resindex]), 
                                            np.sort(nbrs)))
                chids_resnums = []
                environment = []
                env_idxs = []
                chid0 = res_chids[_env_idxs[0]]
                resnum0 = res_resnums[_env_idxs[0]]
                for i, scrr in enumerate(zip(res_segs[_env_idxs], 
                                             res_chids[_env_idxs], 
                                             res_resnums[_env_idxs], 
                                             res_resnames[_env_idxs])):
                    seg, chid, resnum, resname = scrr
                    if i > 0 and resname not in aas:
                        continue
                    d_resnum = np.abs(resnum - resnum0)
                    if chid != chid0 or not d_resnum or \
                            d_resnum >= min_seq_sep:
                        if _env_idxs[i] not in env_idxs:
                            chids_resnums.append((chid, resnum))
                            env_idxs.append(_env_idxs[i])
                            if i > 0:
                                environment.append((biounit, seg, 
                                                    chid, resnum))
                            else:
                                environment.append((biounit, seg, chid, 
                                                    resnum, cg_idx))
                if len(chids_resnums) < 2:
                    continue # No neighbors left.
                if len(env_idxs) > self.max_nbrs:
                    continue
                env_idxs = np.array(env_idxs)
                if os.path.exists(validation_file):
                    rscc_values, rsr_values, rsrz_values = \
                        extract_residue_validation_values(validation_file, 
                                                          chids_resnums)
                else: # accept all residues if no validation file
                    rscc_values = np.ones(len(chids_resnums))
                    rsr_values = np.zeros(len(chids_resnums))
                    rsrz_values = np.zeros(len(chids_resnums))
                betas = res_betas[env_idxs]
                occs = res_occs[env_idxs]
                resnames = res_resnames[env_idxs]
                phis = res_phis[env_idxs[1:]]
                psis = res_psis[env_idxs[1:]]
                resnames_p1 = res_resnames[env_idxs[1:] + 1]
                phis_p1 = res_phis[env_idxs[1:] + 1]
                psis_p1 = res_psis[env_idxs[1:] + 1]
                resnames_m1 = res_resnames[env_idxs[1:] - 1]
                phis_m1 = res_phis[env_idxs[1:] - 1]
                psis_m1 = res_psis[env_idxs[1:] - 1]
                '''
                print('SUMMARY')
                print('resnums:', res_resnums[env_idxs])
                print('RSCC: ', rscc_values, rscc_values > rscc)
                print('RSR:  ', rsr_values, rsr_values < rsr)
                print('RSRZ: ', rsrz_values, rsrz_values < rsrz)
                print('Beta: ', betas, betas < max_b_factor)
                print('Occs: ', occs, occs > min_occ)
                print('Phis: ', phis_m1, phis, phis_p1)
                print('Psis: ', psis_m1, psis, psis_p1)
                '''
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
                             zip(resnames_m1, phis_m1, psis_m1, 
                                 resnames[1:], phis, psis, 
                                 resnames_p1, phis_p1, psis_p1)]
                    do_continue = False
                    for triplet in ABPLE:
                        if 'n' in triplet:
                            do_continue = True
                    if do_continue:
                        # print('n in ABPLE')
                        continue
                    print('All conditions met.')
                    fingerprint_labels.append(
                        self.get_fingerprint(env_idxs, 
                                             sc_info[ent], 
                                             ABPLE))
                    environments.append(environment)
                else:
                    print('Some conditions not met.')
        self.fingerprint_cols = self.contact_cols + self.ABPLE_cols + \
                                self.relpos_cols # update fingerprint_cols
        return fingerprint_labels, environments

    def get_fingerprint(self, env_idxs, ent_sc_info, res_ABPLE_triplets):
        """Find the True labels of the binary fingerprint of an environment.

        Parameters
        ----------
        env_idxs : np.ndarray
            The indices of the residues in the environment.
        ent_sc_info : dict
            Dictionary containing information about the chain 
            that has been mined.
        res_ABPLE_triplets : list
            List of ABPLE classes for each residue in the environment 
            and its neighbors at i - 1 and i + 1.

        Returns
        -------
        true_labels : list
            List of strings for which the binary fingerprint of an 
            environment should be True.
        """
        res_chids = np.array([r.getChid() for r in 
                               ent_sc_info['pdb'].iterResidues()])
        res_resnums = np.array([r.getResnum() for r in 
                                ent_sc_info['pdb'].iterResidues()])
        true_labels = []
        # set the bits corresponding to the contact types
        for env_idx in env_idxs[1:]:
            is_direct = np.logical_and(
                ent_sc_info['nonwater_neighbors'][:, 1] == env_idxs[0], 
                ent_sc_info['nonwater_neighbors'][:, 2] == env_idx, 
            ).sum()
            if is_direct: # direct contact
                # print('IS DIRECT')
                atom_pairs = self.res_contact_to_atom_contacts(
                    env_idxs[0], env_idx, ent_sc_info
                )
                for pair in atom_pairs:
                    # process CG
                    cg_resname = ent_sc_info['pdb'].getResnames()[pair[0]]
                    cg_atomname = ent_sc_info['pdb'].getNames()[pair[0]]
                    cg_atomnames = ent_sc_info['cg_atoms_dict'][cg_resname]
                    # if cg_atomname not in cg_atomnames:
                    #     print(cg_atomname, 'not in CG atoms', cg_atomnames)
                    #     continue
                    if cg_resname in protein_atoms.keys(): # proteinaceous CG
                        if cg_atomname not in protein_atoms[cg_resname]:
                            for el in protein_atoms[cg_resname]:
                                if cg_atomname in el:
                                    cg_atomname = '/'.join(el)
                                    break
                    else: # non-proteinaceous CG; use generic names
                        for match_atomnames in cg_atomnames:
                            cg_resname = 'XXX'
                            if cg_atomname in match_atomnames:
                                cg_atomname = 'atom' + str(
                                    match_atomnames.index(cg_atomname)
                                )
                                break
                        if not cg_atomname.startswith('atom'):
                            continue # contact atom not in SMARTS fragment
                    res_resname = ent_sc_info['pdb'].getResnames()[pair[1]]
                    if res_resname not in protein_atoms.keys():
                        continue
                    res_atomname = ent_sc_info['pdb'].getNames()[pair[1]]
                    if res_atomname not in protein_atoms[res_resname]:
                        for el in protein_atoms[res_resname]:
                            if res_atomname in el:
                                res_atomname = '/'.join(el)
                                break
                    # determine contact type
                    if res_atomname in ['N', 'H', 'CA', 'HA', 'C', 'O']:
                        contact_type = '_'.join([cg_resname, 
                                                 cg_atomname,
                                                 res_atomname])
                    else:
                        contact_type = '_'.join([cg_resname, 
                                                 cg_atomname,
                                                 res_atomname, 
                                                 res_resname])
                    if contact_type not in self.contact_cols:
                        self.contact_cols.append(contact_type)
                    true_labels.append(contact_type)
            bridging_waters = ent_sc_info['water_bridges'][:, 2][
                np.logical_and(
                    ent_sc_info['water_bridges'][:, 1] == 
                        env_idxs[0], 
                    ent_sc_info['water_bridges'][:, 3] == 
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
                        if contact_type not in self.contact_cols:
                            self.contact_cols.append(contact_type)
                        true_labels.append(contact_type)
        # set the bits corresponding to the ABPLE classes
        for i, res_ABPLE_triplet in enumerate(res_ABPLE_triplets):
            idx = i * len(ABPLE_triplets) + \
                  ABPLE_triplets.index(res_ABPLE_triplet)
            true_labels.append(self.ABPLE_cols[idx])
        # set the bits corresponding to the relative positions of residues
        for i in range(len(env_idxs) - 2):
            same_chid = res_chids[env_idxs[i + 2]] == \
                        res_chids[env_idxs[i + 1]]
            relative_pos = res_resnums[env_idxs[i + 2]] - \
                           res_resnums[env_idxs[i + 1]]
            if same_chid and relative_pos < 10:
                idx = i * len(relpos) + relative_pos - 1
                true_labels.append(self.relpos_cols[idx])
            elif same_chid:
                idx = i * len(relpos) + 9
                true_labels.append(self.relpos_cols[idx])
            else:
                idx = i * len(relpos) + 10
                true_labels.append(self.relpos_cols[idx])
        return true_labels    

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


