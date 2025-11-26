import os
import numpy as np

def _pick_atom_by_com(atom_candidates, com, biounit, resname, chain, resnum, atom_name):
    """
    Sometimes, multiple CG atoms share the same atom name within a residue, so choose 
    the one closest to COM. If there are no unambiguous CG atoms to compute COM, choose 
    the first candidate.
    """
    if len(atom_candidates) == 1: # no ambiguity
        return atom_candidates[0]

    if com is None: # no unamibiguous atoms to compute COM
        return atom_candidates[0]

    # Find closest atom to COM.
    best_idx, best_dist = None, None
    for idx, cand_atom in enumerate(atom_candidates):
        c = np.asarray(cand_atom.getCoords())
        c = c[0] if c.ndim == 2 else c
        dist = np.linalg.norm(c - com)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx

    # If even the closest atom is too far from the COM (must be an error with the 
    # parent PDB), skip this environment entirely.
    if best_dist is not None and best_dist > 8.0:
        return None

    return atom_candidates[best_idx]

def _exclusive_lock_path(pdb_path: str) -> str:
    return pdb_path + '.lock'

def _try_acquire_lock(lock_path: str) -> bool:
    """
    Create a lock file atomically: succeed only if it does not yet exist.
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def log_warning(msg, logfile):
    with open(logfile, 'a') as f:
        f.write(msg)

def align_coords_sanity_check(align_coords, eps=1e-6):
    # Any row still ~zero?
    if np.any(np.linalg.norm(align_coords, axis=1) < eps):
        return False

    d01 = align_coords[0] - align_coords[1]
    d21 = align_coords[2] - align_coords[1]

    if np.linalg.norm(d01) < eps or np.linalg.norm(d21) < eps:
        return False  # collapsed points

    cross = np.cross(d01, d21)
    if np.linalg.norm(cross) < eps:
        return False  # collinear -> invalid frame

    # Optional: also check e01 + e21
    e01 = d01 / np.linalg.norm(d01)
    e21 = d21 / np.linalg.norm(d21)
    if np.linalg.norm(e01 + e21) < eps:
        return False

    return True

def _resolve_duplicate_ligand_occupancies(atomgroup, pdb_name):
    """
    Resolve duplicate ligand occupancies (3.x): for each occupancy value with >1 atoms,
    pick one atom to keep (closest to COM of ligand atoms), and REMOVE the other atoms
    from the AtomGroup (atom-level pruning, not residue-level).

    If after resolution there are still duplicates or no ligand atoms remain,
    return None so that caller can skip writing.
    """
    def _validate_ligands(ag):
        """
        Return ag if it has at least one ligand atom (3.0 <= occ < 4.0)
        and no duplicate occupancies (rounded to 0.01). Otherwise return None.
        """
        occs = ag.getOccupancies()
        ligand_idx = [i for i, o in enumerate(occs) if 3.0 <= o < 4.0]
        if not ligand_idx: # no ligand atoms remain
            return None

        # Check that each occupancy (rounded) occurs at most once
        rounded_occs = [round(occs[i], 2) for i in ligand_idx]
        seen = set()
        for o in rounded_occs:
            if o in seen: # duplicate ligand occupancy
                return None
            seen.add(o)
        return ag

    occs = atomgroup.getOccupancies()
    coords = atomgroup.getCoords()

    # Identify ligand atoms by occupancy
    ligand_indices = [i for i, o in enumerate(occs) if 3.0 <= o < 4.0]
    if not ligand_indices: # no ligand
        return None

    # COM over all ligand atoms
    com = coords[ligand_indices].mean(axis=0)

    # Group ligand atoms by occupancy
    occ_to_indices = {}
    for idx in ligand_indices:
        o = round(occs[idx], 2)
        occ_to_indices.setdefault(o, []).append(idx)

    # Determine which atom indices to drop (losers in ambiguous occupancy groups)
    loser_atom_indices = set()
    for o, idxs in occ_to_indices.items():
        if len(idxs) <= 1:
            continue

        # Multiple atoms share this occupancy -> ambiguous; resolve via COM
        candidates = [atomgroup[idx] for idx in idxs]
        chosen_atom = _pick_atom_by_com(
            candidates, com, pdb_name, '?', '?', '?', f'occ_{o}')

        # If no acceptable candidate (e.g., best_dist > 8), skip this environment.
        if chosen_atom is None:
            return None

        chosen_idx = chosen_atom.getIndex()
        for idx in idxs:
            if idx != chosen_idx:
                loser_atom_indices.add(idx)

    # If nothing ambiguous, just sanity-check and return original atomgroup
    if not loser_atom_indices:
        return _validate_ligands(atomgroup)

    # Build a filtered AtomGroup that excludes all losing atoms
    keep_indices = [i for i in range(atomgroup.numAtoms())
                    if i not in loser_atom_indices]
    if not keep_indices:
        # All ligand atoms removed while resolving duplicate occupancies
        return None

    filtered = atomgroup[keep_indices]

    # Final validation: ligands present and occupancies unique
    return _validate_ligands(filtered)