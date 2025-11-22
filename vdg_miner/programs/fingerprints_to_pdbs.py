import os
import sys
import time
import pickle
import argparse
import numpy as np
import prody as pr
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from constants import aas, ABPLE_cols, seqdist_cols, \
                                ABPLE_singleton_cols, cg_atoms

# Reasons for failures could be bad parent PDBs (missing density, mislabeled atoms), 
# obabel conversion issues upstream, etc.

total_environments = 0
failed_environments = 0

def log(msg):
    with open(logfile, 'a') as f:
        f.write(msg)

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

def get_atomgroup(environment, pdb_dir, cg, cg_match_dict, 
                  align_atoms):
    biounit = environment[0][0]
    middle_two = biounit[1:3].lower()
    pdb_file = os.path.join(pdb_dir, middle_two, biounit + '.pdb')

    whole_struct = pr.parsePDB(pdb_file)

    scrs = [(tup[1], tup[2], '`{}`'.format(tup[3])) if tup[3] < 0 else 
            (tup[1], tup[2], tup[3]) for tup in environment]
    selstr_template = '(segment {} and chain {} and resnum {})'
    selstr_template_noseg = '(chain {} and resnum {})'
    selstrs = [selstr_template.format(*scr) if len(scr[0]) else
               selstr_template_noseg.format(*scr[1:]) for scr in scrs]
    sel = whole_struct.select(
        'same residue as within 5 of ({})'.format(' or '.join(selstrs[1:])))
    if sel is None: # neighborhood selection empty; skip environment
        return None, None, None
    struct = sel.toAtomGroup()
    resnames = []
    align_coords = np.zeros((3, 3))

    # Track how many distinct residues we actually map each environment SCR to.
    for i, (scr, selstr) in enumerate(zip(scrs, selstrs)):
        try:
            substruct = struct.select(selstr)
            if substruct is None: # selection empty; skip env
                return None, None, None
            resnames.append(substruct.getResnames()[0])

            # Count unique residue indices for this SCR selection to detect duplication.
            unique_res_indices = np.unique(substruct.getResindices())
            if len(unique_res_indices) != 1:
                log(f'[WARNING] Ambiguous residue selection in {biounit} chain {scr[1]} '
                    f'resnum {scr[2]} (maps to >1 residue); skipping environment.\n')
                return None, None, None

            if i == 0:
                if cg in cg_atoms.keys():
                    atom_names_list = cg_atoms[cg][resnames[0]]
                else:
                    key = (biounit, scrs[0][0], scrs[0][1],
                           str(scrs[0][2]), resnames[0])

                    match_list = cg_match_dict.get(key)
                    match_idx = environment[0][4] - 1  # 1-based index in env --> 0-based

                    if match_list is None: # no CG match; possibly missing density; skip
                        return None, None, None

                    if not (0 <= match_idx < len(match_list)): # out of range; possibly 
                                                               # obabel issue; skip
                        return None, None, None

                    atom_names_list = match_list[match_idx]
 
                # Two-pass selection for CG atoms with ambiguity (a PDB with >2 atoms 
                # of the same CG atom name in the same residue)
                cg_atom_selstrs = ['name ' + atom_name for atom_name in atom_names_list]

                # First pass: collect selections and record non-ambiguous atoms
                sel_list = []
                unambig_atoms = []  # atoms with a single unique match

                for j, cg_selstr in enumerate(cg_atom_selstrs):
                    atom_sel = substruct.select(cg_selstr)

                    if atom_sel is None or atom_sel.numAtoms() == 0: # no atoms; skip
                        return None, None, None

                    sel_list.append(atom_sel)
                    if atom_sel.numAtoms() == 1:
                        unambig_atoms.append(atom_sel[0])

                # Compute COM over all unambiguous CG atoms in case they're needed for 
                # disambiguation of atom names belonging to >1 atom
                com = None
                if len(unambig_atoms) > 0:
                    coords_list = []
                    for a in unambig_atoms:
                        c = np.asarray(a.getCoords())
                        # ProDy may return shape (3,) or (1, 3); normalize
                        c = c[0] if c.ndim == 2 else c
                        coords_list.append(c)
                    com = np.mean(coords_list, axis=0)

                for j, atom_sel in enumerate(sel_list):
                    atom_name = atom_names_list[j]
                    candidates = [atom for atom in atom_sel]
                    resname = resnames[0]
                    chain = scrs[0][1]
                    resnum = scrs[0][2]

                    chosen_atom = _pick_atom_by_com(
                        candidates, com, biounit, resname, chain, resnum, atom_name)

                    # If COM check failed (e.g., best_dist > 8 Å), skip this environment.
                    if chosen_atom is None:
                        return None, None, None

                    # Set the CG-encoding occupancy for this chosen atom
                    chosen_atom.setOccupancy(3.0 + j * 0.1)

                    # Fill alignment coordinates for the chosen CG atoms
                    if j in align_atoms:
                        c = np.asarray(chosen_atom.getCoords())
                        c = c[0] if c.ndim == 2 else c
                        align_coords[align_atoms.index(j)] = c

            else:
                # Non-CG residues: mark them differently
                substruct.setOccupancies(2.0)

        except Exception as e:
            # Environment skipped due to exception in selection processing.
            return None, None, None

    # Build local frame from align_coords
    if not align_coords_sanity_check(align_coords):  # returns T or F
        # Environment skipped: degenerate local frame.
        return None, None, None

    d01 = align_coords[0] - align_coords[1]
    d21 = align_coords[2] - align_coords[1]
    e01 = d01 / np.linalg.norm(d01)
    e21 = d21 / np.linalg.norm(d21)
    e1 = (e01 + e21) / np.linalg.norm(e01 + e21)
    e3 = np.cross(e01, e21) / np.linalg.norm(np.cross(e01, e21))
    e2 = np.cross(e3, e1)
    R = np.array([e1, e2, e3])
    t = align_coords[1]
    coords_transformed = np.dot(struct.getCoords() - t, R.T)
    struct.setCoords(coords_transformed)
    return struct, resnames, whole_struct

def parse_args():
    argp = argparse.ArgumentParser('Generate hierarchy of CG environments.')
    argp.add_argument('-c', '--cg', type=str, required=True,
                      help='Chemical group for which to generate a hierarchy.')
    argp.add_argument('-p', '--pdb-dir', type=str, required=True,
                      help='Path to directory containing PDB files in '
                           'subdirectories named for the middle two '
                           'characters of the PDB ID.')
    argp.add_argument('-f', '--fingerprints-dir', type=str, required=True, 
                      help='Path to directory containing fingerprints.')
    argp.add_argument('-m', '--cg-match-dict-pkl', type=str, 
                      help="Path to the pickled CG match dictionary if "
                           "the CG is not proteinaceous.")
    argp.add_argument('-o', '--output-dir', type=str,
                        help='Path to output dir.')
    argp.add_argument('-s', '--abple-singlets', action='store_true',
                      help='Use ABPLE singlets instead of triplets in the '
                      'hierarchy.')
    argp.add_argument('-e', '--exclude-seqdist', action='store_true', 
                      help='Exclude levels based upon sequence distances '
                           'between contacting residues from the hierarchy.')
    argp.add_argument('-l', "--logfile", default="log", 
                      help="Path to log file.")
    argp.add_argument('-j', '--job-index', type=int, default=0,
                      help='Index for current job (Default: 0).')
    argp.add_argument('-n', '--num-jobs', type=int, default=4,
                      help='Total number of jobs (Default: 1).')
    return argp.parse_args()

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

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    out_dir = args.output_dir
    out_dir = os.path.join(out_dir, 'vdg_pdbs')
    logfile = args.logfile

    written_by_this_job = 0
    
    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # Prevent accidental overwriting if single-job (no sharding)
    if args.num_jobs == 1:
        if os.listdir(out_dir):
            raise ValueError(
                f'The output directory {out_dir} is not empty. Please remove its '
                'contents or specify a new output dir to prevent accidental overwriting.')

    align_atoms = [1, 0, 2] # arbitrary, b/c they'll be re-aligned in clustering
    with open(os.path.join(args.fingerprints_dir, 
                           'fingerprint_cols.txt'), 'r') as f:
        fingerprint_cols = np.array(f.read().split(', '))
    if args.cg_match_dict_pkl is not None:
        with open(args.cg_match_dict_pkl, 'rb') as f:
            cg_match_dict = pickle.load(f)
    else:
        cg_match_dict = {}

    # Build a deterministic, sorted list of input files and shard by file.
    all_fp_files = []
    for subdir in sorted(os.listdir(args.fingerprints_dir)):
        if '.txt' in subdir:
            continue
        full = os.path.join(args.fingerprints_dir, subdir)
        if not os.path.isdir(full):
            continue
        for file in sorted(os.listdir(full)):
            if file.endswith('_fingerprints.npy'):
                all_fp_files.append((subdir, file))
    
    # Shard work: only handle files where idx % num_jobs == job_index
    sharded_fp_files = [
        (subdir, file)
        for idx, (subdir, file) in enumerate(all_fp_files)
        if idx % max(1, args.num_jobs) == args.job_index]

    for subdir, file in sharded_fp_files:
        fingerprint_array = np.load(
            os.path.join(args.fingerprints_dir, subdir, file)
        )
        with open(
                os.path.join(
                    args.fingerprints_dir, 
                    subdir, 
                    file.replace(
                        '_fingerprints.npy', 
                        '_environments.txt')
                    ), 
                    'r'
                ) as f:
            for line, fingerprint in zip(f.readlines(), 
                                         fingerprint_array):
                if len(fingerprint) != len(fingerprint_cols):
                    total_environments += 1
                    failed_environments += 1
                    continue

                environment = eval(line.strip())
                pdb_name = '_'.join([str(el) for el in environment[0]])
                total_environments += 1 # count environment attempt
                atomgroup, resnames, _ = \
                    get_atomgroup(environment, 
                                  args.pdb_dir, cg=args.cg, 
                                  cg_match_dict=cg_match_dict,
                                  align_atoms=align_atoms)

                if atomgroup is None: # skipped for some reason in get_atomgroup
                    failed_environments += 1
                    continue

                # Resolve duplicate ligand occupancies using COM; skip env if it
                # cannot be resolved into a clean CG encoding.
                atomgroup = _resolve_duplicate_ligand_occupancies(atomgroup, pdb_name)
                if atomgroup is None:
                    failed_environments += 1
                    continue

                features = fingerprint_cols[fingerprint]
                features_no_contact = \
                    [feature for feature in features 
                     if feature[:3] != 'XXX' 
                     or feature[:3] not in aas]
                current_res = 1
                dirs = [resnames[current_res]]
                while True:
                    if dirs[-1] in aas:
                        ABPLE = [feature for feature in 
                                 features_no_contact 
                                 if feature in ABPLE_cols and 
                                 feature[0] == str(current_res)]
                        if len(ABPLE):
                            if args.abple_singlets:
                                dirs.append(ABPLE[0].split('_')[0] + '_' + 
                                            ABPLE[0].split('_')[1][1])
                            else:
                                dirs.append(ABPLE[0])
                        else:
                            break
                    elif dirs[-1] in ABPLE_cols or \
                            dirs[-1] in ABPLE_singleton_cols:
                        seqdist = [feature for feature in 
                                   features_no_contact 
                                   if feature in seqdist_cols and 
                                   feature[0] == str(current_res)]
                        if args.exclude_seqdist and len(seqdist):
                            dirs.append('seqdist_any')
                        elif not args.exclude_seqdist and len(seqdist):
                            dirs.append('seqdist_' + seqdist[0][4:])
                        else:
                            break
                    elif 'seqdist' in dirs[-1]:
                        current_res += 1
                        if len(resnames) >= current_res:
                            dirs.append(resnames[current_res])
                        else:
                            dirs.append('no_more_residues')
                            break
                    else:
                        raise ValueError('Invalid feature: ', dirs[-1])
                
                pdb_path = os.path.join(out_dir, f'{pdb_name}.pdb.gz')
                lock_path = _exclusive_lock_path(pdb_path)
                acquired = _try_acquire_lock(lock_path)
                if not acquired:
                    # Another worker is already writing this file (or it exists): skip.
                    continue
                try:
                    final_path = os.path.join(out_dir, f"{pdb_name}.pdb.gz")
                    tmp_path = os.path.join(out_dir, f"{pdb_name}.tmp.{os.getpid()}.pdb.gz")
                    try:
                        pr.writePDB(tmp_path, atomgroup)
                        os.replace(tmp_path, final_path)
                        written_by_this_job += 1
                    except Exception:
                        if os.path.exists(tmp_path): 
                            os.remove(tmp_path)
                        failed_environments += 1 # i/o error
                        log(f'[WARNING] I/O error while writing output file: {pdb_path}\n')
                        continue

                finally:
                    _release_lock(lock_path)
    
    # After processing all environments, report aggregate failure statistics.
    if total_environments > 0:
        failure_pct = 100.0 * failed_environments / total_environments
    else:
        failure_pct = 0.0

    summary_msg = (
        f'\t# environments attempted (job {args.job_index}): {total_environments}\n'
        f'\t# environments skipped   (job {args.job_index}): {failed_environments} '
        f'({failure_pct:.2f}% failures)\n')

    stats_dir = os.path.dirname(logfile)  # e.g., the logs/ directory
    stats_path = os.path.join(stats_dir, f'fp2pdb_stats_job_{args.job_index}.txt')
    with open(stats_path, 'w') as sf:
        # Format: "<total_environments> <failed_environments>\n"
        sf.write(f'{total_environments} {failed_environments}\n')