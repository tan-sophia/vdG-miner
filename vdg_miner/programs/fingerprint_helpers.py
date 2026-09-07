import os
import warnings
import numpy as np

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

def _resolve_duplicate_ligand_occupancies(atomgroup, pdb_name, logfile=None):
    """Return a clean CG encoding, or skip an ambiguous one without guessing.

    CG slots are encoded as occupancies in ``[3.00, 4.00)``.  Two atoms with the
    same slot cannot be distinguished reliably after environment assembly.  The
    former center-of-mass heuristic silently chose one and changed the record;
    reject the whole environment instead.  This function deliberately does not
    alter the AtomGroup, including ProDy's already-selected alternate atoms.
    """
    occs = atomgroup.getOccupancies()
    slots = {}
    for idx, occupancy in enumerate(occs):
        if 3.0 <= occupancy < 4.0:
            slots.setdefault(round(occupancy, 2), []).append(idx)

    if not slots:
        return None

    duplicate_slots = {slot: indices for slot, indices in slots.items()
                       if len(indices) > 1}
    if duplicate_slots:
        details = ', '.join(
            f'{slot:.2f} ({len(indices)} atoms)'
            for slot, indices in sorted(duplicate_slots.items()))
        message = (
            f'[WARNING] {pdb_name}: duplicate CG slot occupancy {details}; '
            'ambiguous ligand atom encoding, skipping environment.\n')
        if logfile is None:
            warnings.warn(message.rstrip(), RuntimeWarning, stacklevel=2)
        else:
            log_warning(message, logfile)
        return None

    return atomgroup
