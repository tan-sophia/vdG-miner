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

class VDG:
    def __init__(self, cg, pdb_dir, cg_natoms=None, min_contact_area=None):
        if cg in cg_resnames.keys():
            self.cg_resnames = cg_resnames[cg]
            self.cg_atoms = {res: cg_atoms[cg][res] for res in cg_resnames[cg]}
        else:
            assert cg_natoms is not None
            self.cg_resnames = ['XXX']
            self.cg_atoms = {'XXX': [f'atom{i}' for i in range(cg_natoms)]}
        self.pdb_dir = pdb_dir
        self.min_contact_area = (sasa.MIN_CONTACT_AREA if min_contact_area is None
                                 else float(min_contact_area))
        self.prev_pdb_file = ''
        self.prev_pdb = None

    def _parse(self, pdb_file):
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
        segnames = pdb.getSegnames()
        chids = pdb.getChids()
        resnums = pdb.getResnums()
        resnames = pdb.getResnames()
        names = pdb.getNames()
        resindices = pdb.getResindices()
        if 'XXX' in self.cg_atoms:
            for key in sorted(k for k in cg_match_dict if k[0] == struct_name):
                _, seg, chain, resnum, resname = key
                try:
                    resnum_match = resnums == int(resnum)
                except (TypeError, ValueError):
                    resnum_match = resnums.astype(str) == str(resnum)
                sel = np.logical_and.reduce((segnames == seg, chids == chain,
                                             resnum_match, resnames == resname))
                if not sel.any():
                    continue
                cg_resindices = tuple(int(r) for r in np.unique(resindices[sel]))
                for j, atom_names in enumerate(cg_match_dict[key]):
                    atom_sel = np.logical_and(sel, np.isin(names, list(atom_names)))
                    idxs = np.flatnonzero(atom_sel)
                    if len(idxs):
                        yield j + 1, cg_resindices, idxs
        else:
            for resname, atom_names in self.cg_atoms.items():
                sel = np.logical_and(resnames == resname, np.isin(names, list(atom_names)))
                if not sel.any():
                    continue
                for resindex in np.unique(resindices[sel]):
                    idxs = np.flatnonzero(np.logical_and(sel, resindices == resindex))
                    yield 1, (int(resindex),), idxs

    def structure_contacts(self, pdb_file, cg_match_dict=None):
        pdb = self._parse(pdb_file)
        struct_name = parent_db.stem_of(pdb_file)
        if 'XXX' in self.cg_atoms:
            cg_atoms_dict = {}
            for key, val in (cg_match_dict or {}).items():
                if key[0] != struct_name:
                    continue
                seen = cg_atoms_dict.setdefault(key[4], [])
                for atom_list in val:
                    if atom_list not in seen:
                        seen.append(atom_list)
        else:
            cg_atoms_dict = {key: [val] for key, val in self.cg_atoms.items()}

        resindices = pdb.getResindices()
        resnames = pdb.getResnames()
        names = pdb.getNames()
        elements = pdb.getElements()
        heavy = np.ones(pdb.numAtoms(), dtype=bool)
        hydrogens = pdb.select('hydrogen')
        if hydrogens is not None:
            heavy[hydrogens.getIndices()] = False
        heavy_idx = np.flatnonzero(heavy)
        heavy_elems = sasa.elements_for(
            names[heavy_idx], None if elements is None else elements[heavy_idx],
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
                    d = self._min_heavy_dist(heavy_coords, heavy_res, cg_heavy, partner)
                    contact_strength[key] = sasa.ResidueContact(0.0, 0.0, 0,
                                                                0, d)
        nonwater_neighbors = (np.unique(np.array(nonwater_neighbors,
                                                 dtype=np.int64), axis=0)
                              if nonwater_neighbors
                              else np.empty((0, 3), dtype=np.int64))
        water_bridges = (np.array(water_bridges, dtype=np.int64)
                         if water_bridges else np.empty((0, 4), dtype=np.int64))

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
        other = np.flatnonzero(heavy_res == resindex)
        if not len(other) or not len(cg_heavy):
            return float('inf')
        d = np.linalg.norm(heavy_coords[cg_heavy][:, None, :] -
                           heavy_coords[other][None, :, :], axis=2)
        return float(d.min())

    def _water_bridges(self, pdb, heavy_idx, heavy_coords, heavy_elems, tree,
                       cg_heavy, cg_resindices, water_resindices):
        if not water_resindices or tree is None:
            return
        heavy_res = pdb.getResindices()[heavy_idx]
        heavy_resnames = pdb.getResnames()[heavy_idx]
        polar = np.array([e in sasa.POLAR_ELEMENTS for e in heavy_elems])
        is_water = np.isin(heavy_res, list(water_resindices))
        cg_polar = np.asarray([i for i in cg_heavy if polar[i]], dtype=np.int64)
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
        pdb = struct['pdb']
        rmask = np.logical_and(struct['res_segnames'] == segi,
                               struct['res_chids'] == chain)
        nonwater = struct['nonwater_neighbors']
        bridges = struct['water_bridges']
        sc_info['_'.join([struct['struct_name'], segi, chain])] = {
            'pdb': pdb,
            'cg_atoms_dict': struct['cg_atoms_dict'],
            'rmask': rmask,
            'nonwater_neighbors': nonwater[rmask[nonwater[:, 1]]],
            'water_bridges': bridges[rmask[bridges[:, 1]]],
            'contact_strength': struct['contact_strength'],
            'res_max_b': struct['res_max_b'],
            'res_min_occ': struct['res_min_occ'],
        }
    def mine_environments(self, chain_cluster=None, cg_match_dict=None,
                          pdb_gz=False, min_seq_sep=1,
                          max_b_factor=100.0, min_occ=0.3,
                          include_non_aa_partners=False):
        sc_info = {}
        pdb_suffix = '.pdb'
        if pdb_gz:
            pdb_suffix += '.gz'
        parsed = {}

        def _structure(pdb_file):
            if pdb_file not in parsed:
                parsed[pdb_file] = self.structure_contacts(pdb_file,
                                                           cg_match_dict)
            return parsed[pdb_file]

        if chain_cluster is not None:
            for mem in chain_cluster:
                biounit = '_'.join(mem.split('_')[:-2])
                assert biounit[4:13] == '_biounit_'
                segi, chain = mem.split('_')[-2:]
                middle_two = biounit[1:3].lower()
                pdb_file = os.path.join(self.pdb_dir, middle_two, biounit + pdb_suffix)
                struct = _structure(pdb_file)
                if struct is not None:
                    self.update_sc_info(sc_info, segi, chain, struct)
        elif cg_match_dict is not None:
            for key in sorted({key[:3] for key in cg_match_dict.keys()}):
                struct_name, segi, chain = key
                pdb_file = parent_db.structure_path(self.pdb_dir, struct_name)
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
            if '__' in ent:
                selstr = 'chain {}'.format(ent.split('_')[-1])
            else:
                selstr = 'segname {} and chain {}'.format(ent.split('_')[-2],
                                                          ent.split('_')[-1])
            selstr += ' and (resname {})'.format(
                ' or resname '.join(sc_info[ent]['cg_atoms_dict']))
            sel = pdb.select(selstr)
            if sel is None:
                continue
            nonwater_neighbors = sc_info[ent]['nonwater_neighbors']
            water_bridges = sc_info[ent]['water_bridges']
            contact_strength = sc_info[ent]['contact_strength']
            unique_cg_idxs = np.unique(np.hstack([nonwater_neighbors[:, 0],
                                                  water_bridges[:, 0]]))
            unique_resindices = np.unique(sel.getResindices())
            for cg_idx in unique_cg_idxs[unique_cg_idxs > 0]:
                for resindex in unique_resindices:
                    nw_mask = np.logical_and(nonwater_neighbors[:, 0] == cg_idx,
                                             nonwater_neighbors[:, 1] == resindex)
                    wb_mask = np.logical_and(water_bridges[:, 0] == cg_idx,
                                             water_bridges[:, 1] == resindex)
                    nbrs = np.concatenate((nonwater_neighbors[nw_mask][:, 2],
                                           water_bridges[wb_mask][:, 3]))
                    _env_idxs = np.concatenate(([resindex], np.sort(nbrs)))
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
                        if i > 0 and not include_non_aa_partners and resname not in aas:
                            continue
                        d_resnum = np.abs(resnum - resnum0)
                        if chid != chid0 or not d_resnum or \
                                d_resnum >= min_seq_sep:
                            if _env_idxs[i] not in env_idxs:
                                chids_resnums.append((chid, resnum))
                                env_idxs.append(_env_idxs[i])
                                if i > 0:
                                    environment.append((biounit, seg, chid, resnum))
                                else:
                                    environment.append((biounit, seg, chid, resnum, cg_idx))
                    if len(chids_resnums) < 2:
                        continue
                    env_idxs = np.array(env_idxs)
                    cg_atom_mask = np.zeros(len(resindices), dtype=bool)
                    cg_names = self._cg_match_atom_names(cg_match_dict, biounit,
                                                         environment[0], cg_idx)
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

                    worst_b = max(cg_max_b, vdm_max_b)
                    worst_occ = min(cg_min_occ, vdm_min_occ)
                    if worst_b >= max_b_factor or worst_occ <= min_occ:
                        continue
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
        if cg_match_dict is None:
            return None
        _biounit, seg, chid, resnum = cg_entry[:4]
        for resnum_key in (str(resnum), resnum):
            for match_key, matches in cg_match_dict.items():
                if (match_key[0] == biounit and match_key[1] == seg
                        and match_key[2] == chid
                        and str(match_key[3]) == str(resnum_key)):
                    if 1 <= cg_idx <= len(matches):
                        return matches[cg_idx - 1]
                    return None
        return None
