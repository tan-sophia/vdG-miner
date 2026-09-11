import os
import sys
import gzip
import numpy as np
import numba as nb
import prody as pr
from ligand_vdgs.functions import parent_db

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from constants import *


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
        List of sliced ATOM and HETATM lines from a PDB file, represented
        as an array.
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
    mine_environments(chain_cluster, cg_match_dict=None, pdb_gz=False,
                      min_seq_sep=1, max_b_factor=100.0, min_occ=0.3)
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
        self.pdb_dir = pdb_dir
        self.probe_dir = probe_dir
        self.validation_dir = validation_dir
        # convenience attributes to prevent unnecessary file reads
        self.prev_pdb_file = ''
        self.prev_pdb = None
        self.prev_pdb_lines = []

    def structure_contacts(self, pdb_file, probe_file, cg_match_dict=None):
        """Parse one structure and derive every contact in it, once.

        None of this depends on which segment/chain the CG sits in: the probe
        file is per structure, and `preprocess_lines`, `find_neighbors` and the
        water-bridge enumeration all run over the whole file. Chains are carved
        out of the result afterwards by `update_sc_info`, which is the only part
        that needs `(segi, chain)`.

        Returns a dict cached by `mine_environments` for the structure's whole
        set of chains, or None if probe and PDB atoms could not be matched.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file corresponding to the structure.
        probe_file : str
            Path to the probe.gz file encoding the contacts in the structure.
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
            if pdb_file.endswith('.gz'):
                with gzip.open(pdb_file, 'rt') as f:
                    for line in f:
                        if line.startswith('ATOM') or \
                                line.startswith('HETATM'):
                            pdb_lines.append(line)
                    pdb = pr.parsePDBStream(f)
            else:
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
        struct_name = parent_db.stem_of(pdb_file)
        if 'XXX' in self.cg_atoms.keys(): # non-proteinaceous CG
            # Union of every copy's match lists, keyed by resname. This feeds
            # preprocess_lines' probe prefilter, which ORs over
            # (resname, atomname), so it must cover the atoms of *all* copies:
            # keying by resname alone and letting the last copy win drops probe
            # lines for atoms only that copy lacks, losing those contacts before
            # anything downstream can recover them. Copies of one resname can
            # differ -- disorder or obabel perception makes a SMARTS match on
            # one chain and not another.
            # NOT an indexable match list: entries from different copies are
            # concatenated, so positions here do not correspond to cg_idx. The
            # per-copy lookup for that is in the cg_idxs loop below.
            cg_atoms_dict = {}
            for key, val in cg_match_dict.items():
                if key[0] != struct_name:
                    continue
                seen = cg_atoms_dict.setdefault(key[4], [])
                for atom_list in val:
                    if atom_list not in seen:
                        seen.append(atom_list)
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
            return None
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
        # One output row per (neighbor, CG site the neighbor atom belongs to).
        # A ligand can match the CG's SMARTS several times with the matches
        # sharing atoms -- SAH's ribose matches OCCO three ways, all overlapping
        # -- so a contacted atom can belong to more than one site. Each such
        # site is credited: the contact is real evidence for every site the
        # atom is part of, and crediting only one leaves the others with
        # truncated environments (or none at all, if that was their only
        # contact). See docs/pitfalls.md for the double-counting caveat.
        row_idxs, row_cg_idxs = [], []
        non_proteinaceous = 'XXX' in self.cg_atoms.keys()
        for i, atom_idx in enumerate(neighbors[:, 0]):
            an = neighbor_atomnames[i]
            if non_proteinaceous:
                # look up by the full (struct, seg, chain, resnum, resname)
                # key, not just resname, so that distinct ligand copies
                # sharing a resname don't alias onto each other's match
                # list (which desyncs the cg_idx recorded here from the
                # match list re-derived for this exact copy downstream)
                line = pdb_lines[atom_idx]
                line_rn = line[17:20].strip()
                line_seg = line[72:76].strip()
                line_chain = line[21]
                line_resnum = line[22:26].strip()
                match_list = cg_match_dict.get(
                    (struct_name, line_seg, line_chain, line_resnum, line_rn))
            else: # proteinaceous CG
                match_list = cg_atoms_dict.get(neighbor_resnames[i])
            hit_idxs = [] if match_list is None else \
                [j + 1 for j, an_list in enumerate(match_list) if an in an_list]
            if not hit_idxs:  # atom is not part of any CG site
                row_idxs.append(i)
                row_cg_idxs.append(0)
            else:
                for cg_idx in hit_idxs:
                    row_idxs.append(i)
                    row_cg_idxs.append(cg_idx)
        cg_idxs = np.array(row_cg_idxs, dtype=np.int64).reshape(-1, 1)
        resindex_neighbors = np.hstack([cg_idxs, resindex_neighbors[row_idxs]])
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
        water_bridges = []
        if water_sel is not None:
            water_resindices = np.unique(water_sel.getResindices())
            nonwater_water_neighbors = \
                resindex_neighbors[np.isin(resindex_neighbors[:, 2],
                                           water_resindices)]
            water_nonwater_neighbors = \
                resindex_neighbors_wat[np.isin(resindex_neighbors_wat[:, 1],
                                               nonwater_resindices)]
            unique_water_neighbors = \
                np.unique(np.hstack([nonwater_water_neighbors[:, 2],
                                     water_nonwater_neighbors[:, 0]]))
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
        if not len(water_bridges):
            water_bridges = np.empty((0, 4), dtype=np.int64)
        # Per-residue quality over the atoms that actually enter a vdG.
        # Reading each residue's *first* atom instead -- the backbone N in a
        # standard residue -- reports a well-ordered backbone for a disordered
        # sidechain, and says nothing at all about the ligand.
        res_segnames = np.array([r.getSegname() for r in pdb.iterResidues()])
        res_chids = np.array([r.getChid() for r in pdb.iterResidues()])
        resindices = pdb.getResindices()
        n_residues = int(resindices.max()) + 1 if len(resindices) else 0
        heavy = np.ones(pdb.numAtoms(), dtype=bool)
        hydrogens = pdb.select('hydrogen')
        if hydrogens is not None:
            heavy[hydrogens.getIndices()] = False
        res_max_b = np.zeros(n_residues, dtype=np.float32)
        res_min_occ = np.full(n_residues, np.inf, dtype=np.float32)
        if heavy.any():
            np.maximum.at(res_max_b, resindices[heavy],
                          pdb.getBetas()[heavy].astype(np.float32))
            np.minimum.at(res_min_occ, resindices[heavy],
                          pdb.getOccupancies()[heavy].astype(np.float32))
        # A residue with no heavy atoms has nothing to judge; do not let an
        # empty aggregate masquerade as zero occupancy.
        res_min_occ[~np.isfinite(res_min_occ)] = 1.0

        return {
            'struct_name' : struct_name,
            'pdb' : pdb,
            'cg_atoms_dict' : cg_atoms_dict,
            'neighbors' : neighbors,
            'neighbors_hb' : neighbors_hb,
            'nonwater_neighbors' : nonwater_neighbors,
            'water_bridges' : water_bridges,
            'res_segnames' : res_segnames,
            'res_chids' : res_chids,
            'res_max_b' : res_max_b,
            'res_min_occ' : res_min_occ,
        }

    def update_sc_info(self, sc_info, segi, chain, struct):
        """Carve one segment/chain out of a parsed structure into `sc_info`.

        `struct` is a `structure_contacts` result, shared by every chain of that
        structure. Only the masks below depend on `(segi, chain)`.
        """
        pdb = struct['pdb']
        mask = np.logical_and(pdb.getSegnames() == segi,
                              pdb.getChids() == chain)
        rmask = np.logical_and(struct['res_segnames'] == segi,
                               struct['res_chids'] == chain)
        neighbors_masked = struct['neighbors'][mask[struct['neighbors'][:, 0]]]
        neighbors_hb_masked = \
            struct['neighbors_hb'][mask[struct['neighbors_hb'][:, 0]]]
        nonwater_neighbors_masked = \
            struct['nonwater_neighbors'][rmask[struct['nonwater_neighbors'][:, 1]]]
        water_bridges_masked = \
            struct['water_bridges'][rmask[struct['water_bridges'][:, 1]]]
        sc_info['_'.join([struct['struct_name'], segi, chain])] = \
            {
                'pdb' : pdb,
                'cg_atoms_dict' : struct['cg_atoms_dict'],
                'mask' : mask,
                'rmask' : rmask,
                'neighbors' : neighbors_masked,
                'neighbors_hb' : neighbors_hb_masked,
                'nonwater_neighbors' : nonwater_neighbors_masked,
                'water_bridges' : water_bridges_masked,
                'res_max_b' : struct['res_max_b'],
                'res_min_occ' : struct['res_min_occ'],
                'num_contacts' : len(nonwater_neighbors_masked) +
                                 len(water_bridges_masked)
            }


    def mine_environments(self, chain_cluster=None, cg_match_dict=None,
                          pdb_gz=False, min_seq_sep=1,
                          max_b_factor=100.0, min_occ=0.3):
        """Mine contact-defined local environments from PDB files.

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
        pdb_gz : bool, optional
            Whether the PDB files are gzipped. Default: False.
        min_seq_sep : int, optional
            Minimum sequence separation between the CG-containing residue
            and a contacting residue in order for the latter to be mined.
            Meaningful only for a proteinaceous CG, where a near-in-sequence
            contact is a consequence of chain connectivity rather than of
            recognition. A ligand CG is not part of the chain, so there the
            comparison is against the ligand's residue number -- an artifact of
            PDB numbering -- and it fires whenever the ligand shares a chain ID
            with the protein and is numbered just past the chain's end, which is
            the usual convention. Default 1, which disables the filter (as does
            0; `d_resnum == 0` is admitted separately).
        max_b_factor : float, optional
            Loose floor on the worst heavy-atom B-factor across the CG and the
            contacting residues. Deliberately permissive: the measured values
            are returned with each environment so a stricter cut can be applied
            downstream without re-mining.
        min_occ : float, optional
            Loose floor on the lowest heavy-atom occupancy, same reasoning.

        Returns
        -------
        environments : list of dict
            One dict per environment: ``env`` holds the residue tuples in the
            established form ``[(biounit, seg, chain, resnum, cg_idx),
            (biounit, seg, chain, resnum), ...]``, alongside the measured
            ``cg_max_b``/``cg_min_occ`` and ``vdm_max_b``/``vdm_min_occ``.
            Neighbor residues are retained when Probe reports either a direct
            contact or a hydrogen-bond-mediated water bridge.
        """
        sc_info = {} # dictionary of information on segment/chain pairs
        pdb_suffix = '.pdb'
        if pdb_gz:
            pdb_suffix += '.gz'
        #print(('Updating sc_info for cluster '
        #       'of length {}').format(len(chain_cluster)))
        # One parse per structure, shared by all of its chains. The probe file
        # is per structure, so re-reading it per chain repeated the gzip read,
        # preprocess_lines, find_neighbors and the water-bridge enumeration --
        # everything except the two masks update_sc_info applies.
        parsed = {}

        def _structure(pdb_file, probe_file):
            key = (pdb_file, probe_file)
            if key not in parsed:
                parsed[key] = self.structure_contacts(
                    pdb_file, probe_file, cg_match_dict)
            return parsed[key]

        if chain_cluster is not None:
            for mem in chain_cluster:
                # resolve necessary paths
                biounit = '_'.join(mem.split('_')[:-2])
                assert biounit[4:13] == '_biounit_'
                segi, chain = mem.split('_')[-2:]
                middle_two = biounit[1:3].lower()
                struct_name = biounit + '_' + segi + '_' + chain
                pdb_file = os.path.join(self.pdb_dir, middle_two,
                                        biounit + pdb_suffix)
                probe_file = os.path.join(self.probe_dir, middle_two,
                                          struct_name + '.probe.gz')
                struct = _structure(pdb_file, probe_file)
                if struct is not None:
                    self.update_sc_info(sc_info, segi, chain, struct)
        elif cg_match_dict is not None:
            for key in sorted({key[:3] for key in cg_match_dict.keys()}):
                struct_name, segi, chain = key
                pdb_file = parent_db.structure_path(self.pdb_dir, struct_name)
                probe_file = parent_db.probe_path(self.probe_dir, struct_name)
                # Skip this chain, not the structure: bailing out here used to
                # discard every other chain's environments too.
                if not os.path.exists(pdb_file) or \
                        not os.path.exists(probe_file):
                    continue
                struct = _structure(pdb_file, probe_file)
                if struct is not None:
                    self.update_sc_info(sc_info, segi, chain, struct)
        else:
            raise ValueError('Either chain_cluster or cg_match_dict '
                             'must not be None.')
        if not len(sc_info):
            return []
        # Evaluate every (segi, chain) copy of the CG rather than only the
        # best-contacted one. Copies are not verified to be geometrically
        # identical, and redundancy is measured downstream by clustering in
        # clus_and_deduplicate_vdgs.py -- discarding copies here would preempt
        # that with an unverified assumption. See docs/pitfalls.md.
        environments = []
        for ent in sc_info:
            biounit = '_'.join(ent.split('_')[:-2])
            pdb = sc_info[ent]['pdb']
            res_segs = np.array([r.getSegname() for r in pdb.iterResidues()])
            res_chids = np.array([r.getChid() for r in pdb.iterResidues()])
            res_resnums = np.array([r.getResnum() for r in pdb.iterResidues()])
            res_resnames = np.array([r.getResname() for r in pdb.iterResidues()])
            res_max_b = sc_info[ent]['res_max_b']
            res_min_occ = sc_info[ent]['res_min_occ']
            resindices = pdb.getResindices()
            atom_names = pdb.getNames()
            atom_betas = pdb.getBetas()
            atom_occs = pdb.getOccupancies()
            if '__' in ent: # no segment name
                selstr = 'chain {}'.format(ent.split('_')[-1])
            else:
                selstr = 'segname {} and chain {}'.format(ent.split('_')[-2],
                                                          ent.split('_')[-1])
            selstr += ' and (resname {})'.format(
                ' or resname '.join(sc_info[ent]['cg_atoms_dict'].keys())
            )
            sel = pdb.select(selstr)
            if sel is None:
                continue  # this chain has no selectable CG residues; try the next
            nonwater_neighbors = sc_info[ent]['nonwater_neighbors']
            water_bridges = sc_info[ent]['water_bridges']
            unique_cg_idxs = np.unique(np.hstack([nonwater_neighbors[:, 0],
                                                  water_bridges[:, 0]]))
            unique_resindices = np.unique(sel.getResindices())
            water_bridges = sc_info[ent]['water_bridges']
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
                    env_idxs = np.array(env_idxs)
                    # CG quality is measured on the CG's own atoms, which no
                    # filter looked at before: a ligand modelled at half
                    # occupancy is exactly the observation that should not count
                    # as evidence that a geometry is real.
                    cg_atom_mask = np.zeros(len(resindices), dtype=bool)
                    cg_names = self._cg_match_atom_names(
                        cg_match_dict, biounit, environment[0], cg_idx)
                    if cg_names is not None:
                        cg_atom_mask = np.logical_and(
                            resindices == env_idxs[0],
                            np.isin(atom_names, list(cg_names)))
                    if cg_atom_mask.any():
                        cg_max_b = float(atom_betas[cg_atom_mask].max())
                        cg_min_occ = float(atom_occs[cg_atom_mask].min())
                    else:
                        cg_max_b = float(res_max_b[env_idxs[0]])
                        cg_min_occ = float(res_min_occ[env_idxs[0]])
                    vdm_idxs = env_idxs[1:]
                    vdm_max_b = float(res_max_b[vdm_idxs].max()) if len(vdm_idxs) else 0.0
                    vdm_min_occ = float(res_min_occ[vdm_idxs].min()) if len(vdm_idxs) else 1.0

                    # A loose floor only. The measured values ride along on the
                    # environment so a stricter threshold can be applied
                    # downstream without re-mining the PDB.
                    worst_b = max(cg_max_b, vdm_max_b)
                    worst_occ = min(cg_min_occ, vdm_min_occ)
                    if worst_b >= max_b_factor or worst_occ <= min_occ:
                        continue
                    environments.append({
                        'env': environment,
                        'cg_max_b': cg_max_b,
                        'cg_min_occ': cg_min_occ,
                        'vdm_max_b': vdm_max_b,
                        'vdm_min_occ': vdm_min_occ,
                    })
        return environments

    @staticmethod
    def _cg_match_atom_names(cg_match_dict, biounit, cg_entry, cg_idx):
        """Atom names of the CG copy `cg_idx` refers to, or None if unresolved.

        `cg_idx` is 1-based into the match list of that exact ligand copy, which
        is how the contact loop assigned it.
        """
        if cg_match_dict is None:
            return None
        _biounit, seg, chid, resnum = cg_entry[:4]
        for resnum_key in (str(resnum), resnum):
            for key, matches in cg_match_dict.items():
                if key[0] == biounit and key[1] == seg and key[2] == chid \
                        and str(key[3]) == str(resnum_key):
                    if 1 <= cg_idx <= len(matches):
                        return matches[cg_idx - 1]
                    return None
        return None

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
        fingerprint : np.ndarray
            Array of booleans denoting whether or not the condition
            associated with each particular fingerprint label is satisfied
            by the environment.
        """
        res_chids = np.array([r.getChid() for r in
                               ent_sc_info['pdb'].iterResidues()])
        res_resnums = np.array([r.getResnum() for r in
                                ent_sc_info['pdb'].iterResidues()])
        # set the bits corresponding to the contact types
        fingerprint = np.zeros(len(self.fingerprint_cols), dtype=np.bool_)
        for env_idx in env_idxs[1:]:
            is_direct = np.logical_and(
                ent_sc_info['nonwater_neighbors'][:, 1] == env_idxs[0],
                ent_sc_info['nonwater_neighbors'][:, 2] == env_idx,
            ).sum()
            if is_direct: # direct contact
                atom_pairs = self.res_contact_to_atom_contacts(
                    env_idxs[0], env_idx, ent_sc_info
                )
                for pair in atom_pairs:
                    # process CG
                    cg_resname = ent_sc_info['pdb'].getResnames()[pair[0]]
                    cg_atomname = ent_sc_info['pdb'].getNames()[pair[0]]
                    cg_atomnames = ent_sc_info['cg_atoms_dict'][cg_resname]
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
                            if type(el) is tuple and res_atomname in el:
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
                        pass
                        #print('Unknown contact type:', contact_type)
                    else:
                        fp_idx = self.contact_cols.index(contact_type)
                        fingerprint[fp_idx] = True
            bridging_waters = ent_sc_info['water_bridges'][:, 2][
                np.logical_and(
                    ent_sc_info['water_bridges'][:, 1] ==
                        env_idxs[0],
                    ent_sc_info['water_bridges'][:, 3] ==
                        env_idx,
                )
            ]
            if len(bridging_waters): # water bridge
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
                            print('Unknown contact type:', contact_type)
                        else:
                            fp_idx = self.contact_cols.index(contact_type)
                            fingerprint[fp_idx] = True
        # set the bits corresponding to the ABPLE classes
        for i, res_ABPLE_triplet in enumerate(res_ABPLE_triplets):
            idx = i * len(ABPLE_triplets) + \
                  ABPLE_triplets.index(res_ABPLE_triplet)
            fp_idx = self.fingerprint_cols.index(self.ABPLE_cols[idx])
            fingerprint[fp_idx] = True
        # set the bits corresponding to the relative positions of residues
        for i in range(min(len(env_idxs) - 2, self.max_nbrs - 1)):
            same_chid = res_chids[env_idxs[i + 2]] == \
                        res_chids[env_idxs[i + 1]]
            relative_pos = res_resnums[env_idxs[i + 2]] - \
                           res_resnums[env_idxs[i + 1]]
            if same_chid and relative_pos < 10:
                idx = i * len(relpos) + relative_pos - 1
                try:
                    fp_idx = self.fingerprint_cols.index(self.relpos_cols[idx])
                except:
                    print('Index error:', i, relative_pos, idx, len(self.relpos_cols))
                    return None
                fingerprint[fp_idx] = True
            elif same_chid:
                idx = i * len(relpos) + 9
                fp_idx = self.fingerprint_cols.index(self.relpos_cols[idx])
                fingerprint[fp_idx] = True
            else:
                idx = i * len(relpos) + 10
                fp_idx = self.fingerprint_cols.index(self.relpos_cols[idx])
                fingerprint[fp_idx] = True
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


def redefine_central_res_if_n(central_res, u):
    # redefine the central res if the central res is n.
    # else, return what it actually is.
    if central_res == 'n':
        non_n = u.replace('n', '')
        if len(non_n) == 0:
            non_n = 'E' # randomly choose an assignment. it doesn't matter for lig vdgs.
        central_res = non_n
    return central_res

def handle_chainbreaks(ABPLE):

    # if ABPLE designation is "n" (likely because of a chain break), replace
    # with the central residue ABPLE designation. if the central residue itself
    # is "n", replace with the only ABPLE designation that's not "n".
    unreplaced_abple = ABPLE
    ABPLE = [] # reinitialize
    for u in unreplaced_abple:
        minus1_res, central_res, plus1_res = u
        if minus1_res == 'n':
            central_res = redefine_central_res_if_n(central_res, u)
            minus1_res = central_res
        if plus1_res == 'n':
            central_res = redefine_central_res_if_n(central_res, u)
            plus1_res = central_res
        replaced_str = minus1_res + central_res + plus1_res
        ABPLE.append(replaced_str)
    return ABPLE
