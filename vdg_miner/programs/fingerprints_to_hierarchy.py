import os
import sys
import gzip
import pickle
import argparse
import numpy as np
import prody as pr

from itertools import product

from vdg_miner.constants import aas, ABPLE_cols, seqdist_cols, \
                                ABPLE_singleton_cols, cg_atoms

def count_files_and_rename_dirs_at_depth(starting_dir, target_depth=1):
    """
    Traverses the directory tree starting from `starting_dir`, counts the
    number of non-directory files in each sub-tree at a specified depth,
    and renames each directory at that depth to include the count of
    non-directory files.

    :param starting_dir: The root directory from which to start the traversal.
    :param target_depth: The depth below which directories should be renamed.
    """
    def get_depth(path):
        return path[len(starting_dir):].count(os.sep)
    for root, dirs, files in os.walk(starting_dir, topdown=False):
        current_depth = get_depth(root)

        if current_depth >= target_depth:
            # Count the number of non-directory files in the current directory
            # and its subdirectories
            num_files = sum([len(files) for _, _, files in os.walk(root)])

            # Get the new directory name with the count of non-directory files
            base_dir = os.path.basename(root)
            parent_dir = os.path.dirname(root)
            new_dir_name = f"{base_dir}_rescount_{num_files}"
            new_dir_path = os.path.join(parent_dir, new_dir_name)

            # Rename the directory
            os.rename(root, new_dir_path)

def create_symlinks_for_pdb_files(starting_dir, target_depth=2):
    """
    Traverse the directory tree and create symlinks for .pdb files in each 
    parent directory, except the directory that contains the .pdb file.

    :param starting_dir: The root directory from which to start the traversal.
    :param target_depth: The depth below which symlinks should be created.
    """
    def get_depth(path):
        return path[len(starting_dir):].count(os.sep)
    for dirpath, _, filenames in os.walk(starting_dir):
        for filename in filenames:
            if filename.endswith('.pdb'):
                pdb_file_path = os.path.join(dirpath, filename)
                parent_path = dirpath
                # Traverse each parent directory except the one containing 
                # the .pdb file
                while parent_path != starting_dir:
                    parent_path = os.path.dirname(parent_path)
                    current_depth = get_depth(parent_path)
                    if current_depth >= target_depth:
                        symlink_path = os.path.join(parent_path, filename)
                        if not os.path.exists(symlink_path):
                            try:
                                os.symlink(pdb_file_path, symlink_path)
                            except Exception as e:
                                error = ("Failed to create symlink: {} -> {}, "
                                        "due to: {}")
                                print(error.format(symlink_path, 
                                                   pdb_file_path, e))

def get_atomgroup(environment, pdb_dir, cg, cg_match_dict, 
                  align_atoms=[1, 0, 2], pdb_gz=False, prev_struct=None):
    """Generate a ProDy AtomGroup for a given environment.

    Parameters
    ----------
    environment : list
        List of tuples containing the SCR information for the environment.
    pdb_dir : str
        Path to directory containing PDB files in subdirectories named for the 
        middle two characters of the PDB ID.
    cg : str
        Chemical group for which to generate a hierarchy.
    cg_match_dict : dict
        Dictionary of matching CGs in ligands with keys as tuples of
        (struct_name, seg, chain, resnum, resname) for the ligand and
        values as lists containing the list of atom names for each match
        to the CG. Used for non-protein CGs.
    align_atoms : list, optional
        Indices of three atoms in the chemical group on which to align the 
        environments. Default is [1, 0, 2].
    pdb_gz : bool, optional
        Whether the PDB files are gzipped. Default is False.
    prev_struct : prody.AtomGroup, optional
        AtomGroup of the previous environment. Default is None.

    Returns
    -------
    struct : prody.AtomGroup
        AtomGroup of the environment.
    resnames : list
        List of residue names for the environment.
    whole_struct : prody.AtomGroup
        AtomGroup of the whole structure from which the environment 
        was extracted.
    """
    biounit = environment[0][0]
    middle_two = biounit[1:3].lower()
    pdb_suffix = '.pdb'
    if pdb_gz:
        pdb_suffix += '.gz'
    pdb_file = os.path.join(pdb_dir, middle_two, 
                            #biounit, 
                            biounit + pdb_suffix)
    if prev_struct is None:
        if pdb_gz:
            with gzip.open(pdb_file, 'rt') as f:
                whole_struct = pr.parsePDBStream(f)
        else:
            whole_struct = pr.parsePDB(pdb_file)
    else:
        whole_struct = prev_struct
    selstr_template = '(segment {} and chain {} and resnum {} and icode _)'
    selstr_template_noseg = '(chain {} and resnum {} and icode _)'
    scrs, selstrs, selstrs_nbrs = [], [], []
    for tup in environment:
        if tup[3] >= 0:
            scr = (tup[1], tup[2], tup[3])
        else:
            scr = (tup[1], tup[2], '`{}`'.format(tup[3]))
        scrs.append(scr)
        if tup[3] - 5 < 0 or tup[3] + 5 < 0:
            nbrs_scr = (tup[1], tup[2], 
                        '`{}:{}`'.format(tup[3] - 5, tup[3] + 5))
        else:
            nbrs_scr = (tup[1], tup[2], 
                        '{}:{}'.format(tup[3] - 5, tup[3] + 5))
        if len(scr[0]):
            selstrs.append(selstr_template.format(*scr))
            selstrs_nbrs.append(selstr_template.format(*nbrs_scr))
        else:
            selstrs.append(selstr_template_noseg.format(*scr[1:]))
            selstrs_nbrs.append(selstr_template_noseg.format(*nbrs_scr[1:]))
    struct = whole_struct.select(
        ' or '.join(selstrs[:1] + selstrs_nbrs[1:]) + 
        ' or (same residue as resname HOH within 3 of ({}))'.format(
            ' or '.join(selstrs[:1])
        )
    ).toAtomGroup()
    # struct = whole_struct.select(
    #     'same residue as within 5 of ({})'.format(' or '.join(selstrs[1:]))
    # ).toAtomGroup()
    resnames = []
    for scr, selstr in zip(scrs, selstrs):
        try:
            substruct = struct.select(selstr)
            resnames.append(substruct.getResnames()[0])
            if scr is scrs[0]:
                if cg in cg_atoms.keys():
                    atom_names_list = cg_atoms[cg][resnames[0]]
                else:
                    key = (biounit, scrs[0][0], scrs[0][1], 
                           str(scrs[0][2]).replace('`', ''), resnames[0])
                    '''
                    if key not in cg_match_dict.keys():
                        print(environment, '\n', scrs, '\n', selstrs, '\n', resnames)
                        print(key)
                        print([k for k in cg_match_dict.keys() if 
                               key[0] == k[0]])
                        print([k for k in cg_match_dict.keys()][:10])
                        sys.exit()
                    '''
                    atom_names_list = \
                        cg_match_dict[key][environment[0][4] - 1]
                cg_atom_selstrs = \
                    ['name ' + atom_name 
                     for atom_name in atom_names_list]
                align_coords = np.zeros((3, 3))
                for j, cg_selstr in enumerate(cg_atom_selstrs):
                    atom_sel = substruct.select(cg_selstr)
                    atom_sel.setOccupancies(3.0 + j * 0.1)
                    if j in align_atoms:
                        align_coords[align_atoms.index(j)] = \
                            atom_sel.getCoords()
            else:
                substruct.setOccupancies(2.0)
        except Exception as e:
            print('Bad SCR: ', biounit, scr)
            print('Exception:', e)
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
    argp.add_argument('-a', '--align-atoms', 
                      default=[1, 0, 2], type=int, nargs='+', 
                      help='Indices of three atoms in the chemical group on '
                           'which to align the environments.')
    argp.add_argument('-o', '--output-hierarchy-dir', type=str,
                        help='Path to directory in which to write hierarchy.')
    argp.add_argument('-s', '--abple-singlets', action='store_true',
                      help='Use ABPLE singlets instead of triplets in the '
                      'hierarchy.')
    argp.add_argument('-n', '--no-abple', action='store_true',
                      help='Do not use ABPLE in the hierarchy (assigning '
                      'every residue an ABPLE string of XXX). Overrides '
                      '--abple-singlets.')
    argp.add_argument('-e', '--exclude-seqdist', action='store_true', 
                      help='Exclude levels based upon sequence distances '
                           'between contacting residues from the hierarchy.')
    return argp.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assert len(args.align_atoms) == 3, 'Must provide three align atoms.'
    with open(os.path.join(args.fingerprints_dir, 
                           'fingerprint_cols.txt'), 'r') as f:
        fingerprint_cols = np.array(f.read().split(', '))
    if args.cg_match_dict_pkl is not None:
        with open(args.cg_match_dict_pkl, 'rb') as f:
            cg_match_dict = pickle.load(f)
    else:
        cg_match_dict = {}
    # iterate over subdirectories of the fingerprints directory, 
    # corresponding to the middle two letters of the PDB files 
    # in which environments were found
    for subdir in os.listdir(args.fingerprints_dir):
        if not os.path.isdir(os.path.join(args.fingerprints_dir, subdir)):
            continue
        # iterate over fingerprints/environments files for 
        # biological assemblies within each subdirectory
        fingerprint_files = [
            os.path.join(args.fingerprints_dir, subdir, file) for file in 
            os.listdir(os.path.join(args.fingerprints_dir, subdir))
            if file.endswith('_fingerprints.npy')
        ]
        for fingerprint_file in fingerprint_files:
            fingerprints = np.load(fingerprint_file)
            if fingerprints.shape[1] != len(fingerprint_cols):
                continue # ensure fingerprints have the correct length
            environment_file = fingerprint_file.replace('_fingerprints.npy', 
                                                        '_environments.txt')
            with open(environment_file, 'r') as f:
                environments = [eval(line.strip()) for line in f.readlines()]
            prev_pdb = ''
            # iterate over environments from each biological assembly
            for fingerprint, environment in zip(fingerprints, environments):
                if prev_pdb == environment[0][0]:
                    atomgroup, resnames, whole_struct = \
                        get_atomgroup(environment, 
                                        args.pdb_dir, args.cg, 
                                        cg_match_dict=cg_match_dict,
                                        align_atoms=args.align_atoms, 
                                        prev_struct=whole_struct)
                else:
                    atomgroup, resnames, whole_struct = \
                        get_atomgroup(environment, 
                                        args.pdb_dir, cg=args.cg, 
                                        cg_match_dict=cg_match_dict,
                                        align_atoms=args.align_atoms)
                    prev_pdb = environment[0][0]
                if atomgroup is None:
                    continue
                features = fingerprint_cols[fingerprint]
                features_no_contact = \
                    [feature for feature in features 
                        if feature[:3] != 'XXX' 
                        or feature[:3] not in aas]
                current_res = 1
                dirs = [resnames[current_res]]
                # construct the hierarchy, one directory at a time
                while True:
                    if dirs[-1] in aas:
                        ABPLE = [feature for feature in 
                                 features_no_contact 
                                 if feature in ABPLE_cols and 
                                 feature[0] == str(current_res)]
                        if len(ABPLE):
                            if args.no_abple:
                                dirs.append(ABPLE[0].split('_')[0] + '_XXX')
                            elif args.abple_singlets:
                                dirs.append(ABPLE[0].split('_')[0] + '_' + 
                                            ABPLE[0].split('_')[1][1])
                            else:
                                dirs.append(ABPLE[0])
                        else:
                            break
                    elif dirs[-1] in ABPLE_cols or \
                            dirs[-1] in ABPLE_singleton_cols or \
                            'XXX' in dirs[-1]:
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
                # write the environment to a PDB file
                pdb_name = '_'.join([str(el) for el in environment[0]])
                hierarchy_path = \
                    '/'.join([args.output_hierarchy_dir] + dirs)
                os.makedirs(hierarchy_path, exist_ok=True)
                pdb_path = hierarchy_path + '/' + pdb_name + '.pdb.gz'
                with gzip.open(pdb_path, "wt") as f:
                    pr.writePDBStream(f, atomgroup)
                # with gzip.open(pdb_path, "rt") as f:
                #     _ = pr.parsePDBStream(f)
                # pr.writePDB(pdb_path, atomgroup)
