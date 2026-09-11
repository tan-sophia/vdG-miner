import os
import sys
import gzip
import numpy as np
import prody as pr
from scipy.spatial import cKDTree
from ligand_vdgs.functions import parent_db
from ligand_vdgs.functions import sasa

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from constants import *

# Placeholder for a vdM slot whose contact strength was not recorded. It should
# never be reached for a direct contact; it exists so a water-bridge-only slot, or
# a residue admitted by a future rule the gate did not measure, yields an explicit
# zero rather than shifting every later slot's value by one.
_NO_CONTACT = sasa.ResidueContact(0.0, 0.0, 0, 0, float('inf'))


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



class VDG:
    """
    Class to store a vdG, a cluster of local environments of a chemical group.

    More specifically, a vdG (van der Graph) is a collection of local
    environments consisting of all residues that form contacts (as assessed
    by buried surface area) with a given chemical group (CG), which itself is a collection
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
    def __init__(self, cg, pdb_dir, validation_dir, cg_natoms=None,
                 min_contact_area=None):
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
        self.validation_dir = validation_dir
        # Membership threshold on `sasa.contact_area`. None takes the module
        # default, which is what a production run should use: theta is a property
        # of the calibration, not of the call site.
        self.min_contact_area = (sasa.MIN_CONTACT_AREA if min_contact_area is None
                                 else float(min_contact_area))
        # convenience attributes to prevent unnecessary file reads
        self.prev_pdb_file = ''
        self.prev_pdb = None

    def _parse(self, pdb_file):
        """Parse one structure, reusing the previous parse when it repeats."""
        if self.prev_pdb_file == pdb_file:
            return self.prev_pdb
        if pdb_file.endswith('.gz'):
            with gzip.open(pdb_file, 'rt') as f:
                pdb = pr.parsePDBStream(f)
        else:
            pdb = pr.parsePDB(pdb_file)
        self.prev_pdb_file = pdb_file
        self.prev_pdb = pdb
        return pdb

    def _cg_copies(self, pdb, struct_name, cg_match_dict):
        """Locate every CG copy in the structure.

        Yields ``(cg_idx, cg_resindices, cg_atom_indices)``. ``cg_idx`` is 1-based
        into the match list of that exact ligand copy, which is the convention the
        rest of the pipeline reads it under; 1 for a proteinaceous CG, which has one
        match per residue.

        ``cg_resindices`` is a tuple because ``find_cg_matches`` keys a ligand by
        ``(struct_name, seg, chain, resnum, resname)`` with the insertion code
        dropped, so two residues differing only in icode arrive as one ligand and
        both of their residues have to be treated as the CG's own (they are excluded
        from being partners). That aliasing is upstream in `cg.py`, not fixed here.
        """
        segnames = pdb.getSegnames()
        chids = pdb.getChids()
        resnums = pdb.getResnums()
        resnames = pdb.getResnames()
        names = pdb.getNames()
        resindices = pdb.getResindices()
        if 'XXX' in self.cg_atoms:   # non-proteinaceous CG
            for key in sorted(k for k in cg_match_dict if k[0] == struct_name):
                _, seg, chain, resnum, resname = key
                # `cg.py` writes the resnum as the raw column text; compare as
                # integers so " 066" and 66 are the same residue.
                try:
                    resnum_match = resnums == int(resnum)
                except (TypeError, ValueError):
                    resnum_match = resnums.astype(str) == str(resnum)
                sel = np.logical_and.reduce((
                    segnames == seg, chids == chain, resnum_match,
                    resnames == resname))
                if not sel.any():
                    continue
                cg_resindices = tuple(int(r) for r in np.unique(resindices[sel]))
                for j, atom_names in enumerate(cg_match_dict[key]):
                    atom_sel = np.logical_and(sel, np.isin(names,
                                                           list(atom_names)))
                    idxs = np.flatnonzero(atom_sel)
                    if len(idxs):
                        yield j + 1, cg_resindices, idxs
        else:                         # proteinaceous CG
            for resname, atom_names in self.cg_atoms.items():
                sel = np.logical_and(resnames == resname,
                                     np.isin(names, list(atom_names)))
                if not sel.any():
                    continue
                for resindex in np.unique(resindices[sel]):
                    idxs = np.flatnonzero(np.logical_and(
                        sel, resindices == resindex))
                    yield 1, (int(resindex),), idxs

    def structure_contacts(self, pdb_file, cg_match_dict=None):
        """Parse one structure and derive every CG contact in it, once.

        A residue contacts the CG when `sasa.contact_area` -- its exclusive buried
        area plus its 1/k share of the surface it occludes jointly with another
        residue -- exceeds ``min_contact_area`` A^2. That replaces Probe and any
        distance rule: occlusion is part of the measurement rather than something a
        cutoff has to approximate, and because it is computed on heavy atoms with
        radii fitted to Probe's own output, membership no longer depends on where a
        protonation program put the hydrogens. Distance enters only as the candidate
        prefilter, which is a superset of the gate by construction.

        The shared term is not a refinement: exclusive area alone has a measured
        recall ceiling of 0.9485 against Probe, because a residue whose whole patch
        is covered by a second residue is credited nothing by leave-one-out
        (decision_records.md DR-3).

        None of this depends on which segment/chain the CG sits in; chains are
        carved out of the result afterwards by `update_sc_info`, which is the only
        part that needs ``(segi, chain)``.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file corresponding to the structure.
        cg_match_dict : dict, optional
            Dictionary of matching CGs in ligands with keys as tuples of
            (struct_name, seg, chain, resnum, resname) for the ligand and
            values as lists containing the list of atom names for each match
            to the CG. Used for non-protein CGs. Default: None.
        """
        pdb = self._parse(pdb_file)
        struct_name = parent_db.stem_of(pdb_file)
        if 'XXX' in self.cg_atoms.keys(): # non-proteinaceous CG
            # Union of every copy's match lists, keyed by resname, for the residue
            # selection `mine_environments` builds. NOT an indexable match list:
            # entries from different copies are concatenated, so positions here do
            # not correspond to cg_idx. The per-copy lookup for that is
            # `_cg_copies`.
            cg_atoms_dict = {}
            for key, val in (cg_match_dict or {}).items():
                if key[0] != struct_name:
                    continue
                seen = cg_atoms_dict.setdefault(key[4], [])
                for atom_list in val:
                    if atom_list not in seen:
                        seen.append(atom_list)
        else: # proteinaceous CG
            cg_atoms_dict = {key : [val]
                             for key, val in self.cg_atoms.items()}

        resindices = pdb.getResindices()
        resnames = pdb.getResnames()
        names = pdb.getNames()
        elements = pdb.getElements()
        # Heavy atoms only, as surface and as occluders alike (sasa module
        # docstring). Hydrogens are also what the old Probe path keyed on, so this
        # is the line where placed-hydrogen dependence leaves the pipeline.
        heavy = np.ones(pdb.numAtoms(), dtype=bool)
        hydrogens = pdb.select('hydrogen')
        if hydrogens is not None:
            heavy[hydrogens.getIndices()] = False
        heavy_idx = np.flatnonzero(heavy)
        heavy_elems = sasa.elements_for(names[heavy_idx],
                                        None if elements is None
                                        else elements[heavy_idx],
                                        resnames[heavy_idx])
        heavy_radii = sasa.radii_for(heavy_elems)
        heavy_coords = pdb.getCoords()[heavy_idx]
        heavy_res = resindices[heavy_idx]
        tree = cKDTree(heavy_coords) if len(heavy_coords) else None

        is_water = np.isin(resnames, list(sasa.WATER_RESNAMES))
        water_resindices = set(int(r) for r in np.unique(resindices[is_water]))

        nonwater_neighbors = []
        water_bridges = []
        contact_strength = {}
        theta = self.min_contact_area
        for cg_idx, cg_resindices, cg_atoms in self._cg_copies(
                pdb, struct_name, cg_match_dict or {}):
            if tree is None:
                continue
            cg_heavy = np.flatnonzero(np.isin(heavy_idx, cg_atoms))
            if not len(cg_heavy):
                continue
            cg_resindex = cg_resindices[0]
            contacts = sasa.buried_area_by_residue(
                heavy_coords, heavy_radii, heavy_res, cg_heavy,
                exclude_resindices=cg_resindices, tree=tree)
            for resindex, strength in contacts.items():
                if resindex in water_resindices:
                    continue
                if sasa.contact_area(strength) <= theta:
                    continue
                nonwater_neighbors.append([cg_idx, cg_resindex, resindex])
                contact_strength[(cg_idx, cg_resindex, resindex)] = strength
            for water, partner in self._water_bridges(
                    pdb, heavy_idx, heavy_coords, heavy_elems, tree,
                    cg_heavy, cg_resindices, water_resindices):
                if partner in cg_resindices:
                    continue
                row = [cg_idx, cg_resindex, water, partner]
                if row not in water_bridges:
                    water_bridges.append(row)
                key = (cg_idx, cg_resindex, partner)
                if key not in contact_strength:
                    # Reached only through the water, so it buries no CG surface
                    # and has no atom pair inside the prefilter. The distance is
                    # still real and still worth recording.
                    d = self._min_heavy_dist(heavy_coords, heavy_res, cg_heavy,
                                             partner)
                    contact_strength[key] = sasa.ResidueContact(0.0, 0.0, 0,
                                                                0, d)
        nonwater_neighbors = (np.unique(np.array(nonwater_neighbors,
                                                 dtype=np.int64), axis=0)
                              if nonwater_neighbors
                              else np.empty((0, 3), dtype=np.int64))
        water_bridges = (np.array(water_bridges, dtype=np.int64)
                         if water_bridges else np.empty((0, 4), dtype=np.int64))

        # Per-residue quality over the atoms that actually enter a vdG.
        # Reading each residue's *first* atom instead -- the backbone N in a
        # standard residue -- reports a well-ordered backbone for a disordered
        # sidechain, and says nothing at all about the ligand.
        res_segnames = np.array([r.getSegname() for r in pdb.iterResidues()])
        res_chids = np.array([r.getChid() for r in pdb.iterResidues()])
        n_residues = int(resindices.max()) + 1 if len(resindices) else 0
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
            'nonwater_neighbors' : nonwater_neighbors,
            'water_bridges' : water_bridges,
            'contact_strength' : contact_strength,
            'res_segnames' : res_segnames,
            'res_chids' : res_chids,
            'res_max_b' : res_max_b,
            'res_min_occ' : res_min_occ,
        }

    @staticmethod
    def _min_heavy_dist(heavy_coords, heavy_res, cg_heavy, resindex):
        """Closest heavy-atom distance from the CG to one residue, or inf."""
        other = np.flatnonzero(heavy_res == resindex)
        if not len(other) or not len(cg_heavy):
            return float('inf')
        d = np.linalg.norm(heavy_coords[cg_heavy][:, None, :] -
                           heavy_coords[other][None, :, :], axis=2)
        return float(d.min())

    def _water_bridges(self, pdb, heavy_idx, heavy_coords, heavy_elems, tree,
                       cg_heavy, cg_resindices, water_resindices):
        """Waters relaying the CG to a residue; yields ``(water, partner)``.

        A bridging water is a hydrogen-bond relay, so both legs are polar-atom
        distance gates (N/O...O within `sasa.WATER_BRIDGE_DIST`) rather than buried
        area: the water buries the CG on the way past whether or not it donates, and
        area alone would admit any water that happens to pack against the group.
        Waters are absent from the current parent database, so this path is
        exercised by tests only until they return.
        """
        if not water_resindices or tree is None:
            return
        heavy_res = pdb.getResindices()[heavy_idx]
        heavy_resnames = pdb.getResnames()[heavy_idx]
        polar = np.array([e in sasa.POLAR_ELEMENTS for e in heavy_elems])
        is_water = np.isin(heavy_res, list(water_resindices))
        cg_polar = np.array([i for i in cg_heavy if polar[i]], dtype=np.int64)
        if not len(cg_polar):
            return
        water_o = np.flatnonzero(np.logical_and(is_water, polar))
        if not len(water_o):
            return
        d_cg = np.linalg.norm(heavy_coords[cg_polar][:, None, :] -
                              heavy_coords[water_o][None, :, :], axis=2)
        near_cg = water_o[(d_cg <= sasa.WATER_BRIDGE_DIST).any(axis=0)]
        if not len(near_cg):
            return
        # Partner leg: any polar atom of a non-water residue, but not the CG's own
        # residue and not another water (a water chain is not a bridge).
        partner_pool = np.flatnonzero(np.logical_and.reduce((
            polar, ~is_water, ~np.isin(heavy_res, list(cg_resindices)),
            ~np.isin(heavy_resnames, list(sasa.WATER_RESNAMES)))))
        if not len(partner_pool):
            return
        d_p = np.linalg.norm(heavy_coords[near_cg][:, None, :] -
                             heavy_coords[partner_pool][None, :, :], axis=2)
        hit = d_p <= sasa.WATER_BRIDGE_DIST
        for wi, w in enumerate(near_cg):
            for pi in np.flatnonzero(hit[wi]):
                yield int(heavy_res[w]), int(heavy_res[partner_pool[pi]])

    def update_sc_info(self, sc_info, segi, chain, struct):
        """Carve one segment/chain out of a parsed structure into `sc_info`.

        `struct` is a `structure_contacts` result, shared by every chain of that
        structure. Only the masks below depend on `(segi, chain)`. The contact
        strengths are keyed by ``(cg_idx, cg_resindex, nbr_resindex)``, so they are
        already carved by the same masks and are passed through whole.
        """
        pdb = struct['pdb']
        rmask = np.logical_and(struct['res_segnames'] == segi,
                               struct['res_chids'] == chain)
        nonwater_neighbors_masked = \
            struct['nonwater_neighbors'][rmask[struct['nonwater_neighbors'][:, 1]]]
        water_bridges_masked = \
            struct['water_bridges'][rmask[struct['water_bridges'][:, 1]]]
        sc_info['_'.join([struct['struct_name'], segi, chain])] = \
            {
                'pdb' : pdb,
                'cg_atoms_dict' : struct['cg_atoms_dict'],
                'rmask' : rmask,
                'nonwater_neighbors' : nonwater_neighbors_masked,
                'water_bridges' : water_bridges_masked,
                'contact_strength' : struct['contact_strength'],
                'res_max_b' : struct['res_max_b'],
                'res_min_occ' : struct['res_min_occ'],
            }

    def mine_environments(self, chain_cluster=None, cg_match_dict=None,
                          pdb_gz=False, min_seq_sep=1,
                          max_b_factor=100.0, min_occ=0.3,
                          include_non_aa_partners=False):
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
        include_non_aa_partners : bool, optional
            Admit metal ions and second ligands as partner residues. The gate
            already measures them -- buried area does not care what a residue is --
            so this switch only controls whether they are emitted. Default False
            because a non-canonical partner resname is a contract change for the
            npz writer, not just more rows: the AA-composition bucket label and the
            slot-label scheme have to accept it. Metals are also untestable on the
            current parent database, which has none. Flip it once the writer side
            is agreed and a parent database with ions exists.

        Returns
        -------
        environments : list of dict
            One dict per environment: ``env`` holds the residue tuples in the
            established form ``[(biounit, seg, chain, resnum, cg_idx),
            (biounit, seg, chain, resnum), ...]``, alongside the measured
            ``cg_max_b``/``cg_min_occ`` and ``vdm_max_b``/``vdm_min_occ``, and the
            per-vdM contact strengths ``buried_area``, ``shared_area``,
            ``n_atom_pairs`` and ``min_heavy_dist``, each a list aligned with
            ``env[1:]``. Neighbor residues are retained when they bury CG surface
            area or relay to it through a bridging water.
        """
        sc_info = {} # dictionary of information on segment/chain pairs
        pdb_suffix = '.pdb'
        if pdb_gz:
            pdb_suffix += '.gz'
        #print(('Updating sc_info for cluster '
        #       'of length {}').format(len(chain_cluster)))
        # One parse per structure, shared by all of its chains: the gate runs over
        # the whole structure, so re-deriving it per chain repeats the parse, the
        # KD-tree build and every SASA pass -- everything except the mask
        # update_sc_info applies.
        parsed = {}

        def _structure(pdb_file):
            if pdb_file not in parsed:
                parsed[pdb_file] = self.structure_contacts(pdb_file,
                                                           cg_match_dict)
            return parsed[pdb_file]

        if chain_cluster is not None:
            for mem in chain_cluster:
                # resolve necessary paths
                biounit = '_'.join(mem.split('_')[:-2])
                assert biounit[4:13] == '_biounit_'
                segi, chain = mem.split('_')[-2:]
                middle_two = biounit[1:3].lower()
                pdb_file = os.path.join(self.pdb_dir, middle_two,
                                        biounit + pdb_suffix)
                struct = _structure(pdb_file)
                if struct is not None:
                    self.update_sc_info(sc_info, segi, chain, struct)
        elif cg_match_dict is not None:
            for key in sorted({key[:3] for key in cg_match_dict.keys()}):
                struct_name, segi, chain = key
                pdb_file = parent_db.structure_path(self.pdb_dir, struct_name)
                # Skip this chain, not the structure: bailing out here used to
                # discard every other chain's environments too.
                if not os.path.exists(pdb_file):
                    continue
                struct = _structure(pdb_file)
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
            contact_strength = sc_info[ent]['contact_strength']
            unique_cg_idxs = np.unique(np.hstack([nonwater_neighbors[:, 0],
                                                  water_bridges[:, 0]]))
            unique_resindices = np.unique(sel.getResindices())
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
                        if i > 0 and not include_non_aa_partners and \
                                resname not in aas:
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
                    # One value per vdM slot, in the order of env[1:]. The loop
                    # above sorts and de-duplicates the neighbour resindices, so
                    # row order from `nonwater_neighbors` is gone by here and the
                    # strengths have to be looked up by key -- this alignment is
                    # the hand-off the npz columns are built from.
                    strengths = [
                        contact_strength.get((int(cg_idx), int(env_idxs[0]),
                                              int(r)), _NO_CONTACT)
                        for r in vdm_idxs
                    ]
                    environments.append({
                        'env': environment,
                        'cg_max_b': cg_max_b,
                        'cg_min_occ': cg_min_occ,
                        'vdm_max_b': vdm_max_b,
                        'vdm_min_occ': vdm_min_occ,
                        'buried_area': [s.buried_area for s in strengths],
                        'shared_area': [s.shared_area for s in strengths],
                        'n_atom_pairs': [s.n_atom_pairs for s in strengths],
                        'min_heavy_dist': [s.min_heavy_dist for s in strengths],
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
