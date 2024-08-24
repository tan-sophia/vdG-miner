import os
import sys
import pickle
import argparse
import numpy as np
import prody as pr

from itertools import product

from vdg_miner.constants import aas, ABPLE_cols, seqdist_cols, cg_atoms

def count_non_directory_files(folder_path):
    """
    Count the number of non-directory files in a given folder.
    """
    count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            count += 1
    return count

def rename_folders_with_file_count(root_path):
    """
    Traverse the directory tree and rename each folder to include the count of 
    non-directory files.
    """
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Skip the root directory itself
        if dirpath == root_path:
            continue

        # Get the current folder name and parent path
        current_folder_name = os.path.basename(dirpath)
        parent_path = os.path.dirname(dirpath)

        # Count the non-directory files in the current folder
        file_count = count_non_directory_files(dirpath)

        # Construct the new folder name
        new_folder_name = f"{current_folder_name}_{file_count}"

        # Construct the full new path
        new_folder_path = os.path.join(parent_path, new_folder_name)

        # Rename the folder
        os.rename(dirpath, new_folder_path)

        print(f"Renamed: {dirpath} -> {new_folder_path}")

def get_atomgroup(environment, pdb_dir, cg, cg_match_dict, 
                  align_atoms=[0, 1, 2], prev_struct=None):
    biounit = environment[0][0]
    middle_two = biounit[1:3].lower()
    pdb_file = os.path.join(pdb_dir, middle_two, biounit + '.pdb')
    if prev_struct is None:
        whole_struct = pr.parsePDB(pdb_file)
    else:
        whole_struct = prev_struct
    scrs = [(tup[1], tup[2], '`{}`'.format(tup[3])) if tup[3] < 0 else 
            (tup[1], tup[2], tup[3]) for tup in environment]
    selstr_template = '(segment {} and chain {} and resnum {})'
    selstr_template_noseg = '(chain {} and resnum {})'
    selstrs = [selstr_template.format(*scr) if len(scr[0]) else
               selstr_template_noseg.format(*scr[1:]) for scr in scrs]
    struct = whole_struct.select(
        'same residue as within 5 of ({})'.format(' or '.join(selstrs[1:]))
    ).toAtomGroup()
    cg_res_struct = whole_struct.select(
        '({}) within 5 of ({})'.format(' or '.join(selstrs[:1]), 
                                       ' or '.join(selstrs[1:]))
    ).toAtomGroup()
    resnames = []
    cg_res_atomnames = []
    for i, (scr, selstr) in enumerate(zip(scrs, selstrs)):
        try:
            if i == 0:
                struct.select(selstr).setOccupancies(3.0)
                cg_res_atomnames = \
                    set(cg_res_struct.select(selstr).getNames())
            else:   
                struct.select(selstr).setOccupancies(2.0)
        except:
            print('Bad SCR: ', biounit, scr)
            return None, None, None
    resnames.append(struct.select(selstr).getResnames()[0])
    if cg in cg_atoms.keys():
        align_atom_names = [cg_atoms[cg][resnames[0]][i] 
                            for i in align_atoms]
    else:
        key = (biounit, scrs[0][0], scrs[0][1], str(scrs[0][2]), 
                resnames[0])
        for atom_names_list in cg_match_dict[key]:
            if set(atom_names_list).issubset(cg_res_atomnames):
                break
        align_atom_names = [atom_names_list[i] for i in align_atoms]
    align_coords = np.vstack(
        [struct.select(
            selstrs[0] + ' and name {}'.format(align_atom_name)
            ).getCoords() for align_atom_name in align_atom_names]
        )
    # except:
    #     print('Bad environment to align: ', environment)
    #     return None, None, None
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
                      default=['1', '0', '2'], nargs='+', 
                      help='Indices of three atoms in the chemical group on '
                           'which to align the environments.')
    argp.add_argument('-o', '--output-hierarchy-dir', type=str,
                        help='Path to directory in which to write hierarchy.')
    argp.add_argument('-s', '--abple-singlets', action='store_true',
                      help='Use ABPLE singlets instead of triplets in the '
                      'hierarchy.')
    return argp.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assert len(args.align_atoms) == 3, 'Must provide three align atoms.'
    align_atoms = np.array([int(a) for a in args.align_atoms])
    os.makedirs(args.output_hierarchy_dir, exist_ok=True)
    with open(os.path.join(args.fingerprints_dir, 
                           'fingerprint_cols.txt'), 'r') as f:
        fingerprint_cols = np.array(f.read().split(', '))
    if args.cg_match_dict_pkl is not None:
        with open(args.cg_match_dict_pkl, 'rb') as f:
            cg_match_dict = pickle.load(f)
    else:
        cg_match_dict = {}
    for subdir in os.listdir(args.fingerprints_dir):
        if '.txt' in subdir:
            continue
        for file in os.listdir(os.path.join(args.fingerprints_dir, subdir)):
            if file.endswith('_fingerprints.npy'):
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
                    prev_pdb = ''
                    for line, fingerprint in zip(f.readlines(), 
                                                 fingerprint_array):
                        environment = eval(line.strip())
                        pdb_name = '_'.join([str(el) for el in environment[0]])
                        if prev_pdb == environment[0][0]:
                            atomgroup, resnames, whole_struct = \
                                get_atomgroup(environment, 
                                              args.pdb_dir, args.cg, 
                                              cg_match_dict=cg_match_dict,
                                              align_atoms=align_atoms, 
                                              prev_struct=whole_struct)
                        else:
                            atomgroup, resnames, whole_struct = \
                                get_atomgroup(environment, 
                                              args.pdb_dir, cg=args.cg, 
                                              cg_match_dict=cg_match_dict,
                                              align_atoms=align_atoms)
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
                        while True:
                            if dirs[-1] in aas:
                                ABPLE = [feature for feature in 
                                         features_no_contact 
                                         if feature in ABPLE_cols and 
                                         feature[0] == str(current_res)]
                                if len(ABPLE):
                                    if args.abple_singlets:
                                        dirs.append(ABPLE[0][-2])
                                    else:
                                        dirs.append(ABPLE[0])
                                else:
                                    break
                            elif dirs[-1] in ABPLE_cols:
                                seqdist = [feature for feature in 
                                           features_no_contact 
                                           if feature in seqdist_cols and 
                                           feature[0] == str(current_res)]
                                if len(seqdist):
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
                        hierarchy_path = \
                            '/'.join([args.output_hierarchy_dir[:-1]] + dirs)
                        os.makedirs(hierarchy_path, exist_ok=True)
                        pdb_path = hierarchy_path + '/' + pdb_name + '.pdb'
                        pr.writePDB(pdb_path, atomgroup)
                        '''
                        for i in range(1, len(dirs)):
                            hierarchy_path = '/'.join([hierarchy_dir[:-1]] + 
                                                      dirs[:-i])
                            symlink_path = hierarchy_path + '/' + \
                                           pdb_name + '.pdb'
                            os.symlink(pdb_path, symlink_path)
                        '''
    # rename_folders_with_file_count(hierarchy_dir)

                            


                        
                        
