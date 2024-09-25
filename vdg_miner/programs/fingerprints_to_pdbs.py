import os
import sys
import time
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
                                error_message = error.format(symlink_path, 
                                                pdb_file_path, e)
                                with open(logfile, 'a') as file:
                                    
                                    file.write(error_message + '\n')
                                

def get_atomgroup(environment, pdb_dir, cg, cg_match_dict, 
                  align_atoms=[1, 0, 2], prev_struct=None):
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
    resnames = []
    for i, (scr, selstr) in enumerate(zip(scrs, selstrs)):
        try:
            substruct = struct.select(selstr)
            resnames.append(substruct.getResnames()[0])
            if i == 0:
                if cg in cg_atoms.keys():
                    atom_names_list = cg_atoms[cg][resnames[0]]
                else:
                    key = (biounit, scrs[0][0], scrs[0][1], 
                           str(scrs[0][2]), resnames[0])
                    atom_names_list = \
                        cg_match_dict[key][environment[0][4] - 1]
                cg_atom_selstrs = \
                    ['name ' + atom_name 
                     for atom_name in atom_names_list]
                align_coords = np.zeros((3, 3))
                for j, selstr in enumerate(cg_atom_selstrs):
                    atom_sel = substruct.select(selstr)
                    atom_sel.setOccupancies(3.0 + j * 0.1)
                    if j in align_atoms:
                        align_coords[align_atoms.index(j)] = \
                            atom_sel.getCoords()
            else:
                substruct.setOccupancies(2.0)
        except:
            pass
            #with open(logfile, 'a') as file:
            #    file.write(f'Bad SCR: {biounit} {scr} \n')
            
            
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
    argp.add_argument('-o', '--output-hierarchy-dir', type=str,
                        help='Path to directory in which to write hierarchy.')
    argp.add_argument('-s', '--abple-singlets', action='store_true',
                      help='Use ABPLE singlets instead of triplets in the '
                      'hierarchy.')
    argp.add_argument('-e', '--exclude-seqdist', action='store_true', 
                      help='Exclude levels based upon sequence distances '
                           'between contacting residues from the hierarchy.')
    argp.add_argument('-l', "--logfile", default="log", 
                      help="Path to log file.")
    return argp.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    out_dir = args.output_hierarchy_dir
    out_dir = os.path.join(out_dir, args.cg, 'vdg_pdbs')
    logfile = args.logfile
    
    # Prepare output directory
    if os.path.exists(out_dir):
        if os.listdir(out_dir):
            raise ValueError(f'The output directory {out_dir} is not empty. Please remove '
                             'its contents or specify a new output dir to prevent accidental '
                             'overwriting.')
    else:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(logfile, 'a') as file:
        file.write(f"{'='*15} Starting fingerprints_to_pdbs.py run {'='*15} \n")

    #assert len(args.align_atoms) == 3, 'Must provide three align atoms.'
    align_atoms = [0, 1, 2]
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
                        if len(fingerprint) != len(fingerprint_cols):
                            continue
                        environment = eval(line.strip())
                        pdb_name = '_'.join([str(el) for el in environment[0]])
                        if prev_pdb == environment[0][0]:
                            atomgroup, resnames, whole_struct = \
                                get_atomgroup(environment, 
                                              args.pdb_dir, args.cg, 
                                              cg_match_dict=cg_match_dict,
                                              align_atoms=[0, 1, 2],
                                              prev_struct=whole_struct)
                        else:
                            atomgroup, resnames, whole_struct = \
                                get_atomgroup(environment, 
                                              args.pdb_dir, cg=args.cg, 
                                              cg_match_dict=cg_match_dict,
                                              align_atoms=[0, 1, 2])
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
                        hierarchy_path = \
                            '/'.join([out_dir] + dirs)
                        #os.makedirs(hierarchy_path, exist_ok=True)
                        #pdb_path = hierarchy_path + '/' + pdb_name + '.pdb'
                        
                        # Output all the pdbs to a single directory, instead of the
                        # hierarchical structure.
                        pdb_path = os.path.join(out_dir, f'{pdb_name}.pdb')
                        pr.writePDB(pdb_path, atomgroup)


    # Print out time elapsed
    seconds = time.time() - start_time
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds = round(seconds, 2)
    
    
    with open(logfile, 'a') as file:
        file.write(f"Completed fingerprints_to_pdbs.py in {hours} h, ")
        file.write(f"{minutes} mins, and {seconds} secs \n")


    #count_files_and_rename_dirs_at_depth(out_dir, 1)
    #create_symlinks_for_pdb_files(out_dir, 2)