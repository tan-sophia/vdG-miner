import os
import sys
import numpy as np
import prody as pr

from scipy.spatial.distance import cdist

from itertools import product

aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
       'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
       'SER', 'THR', 'TRP', 'TYR', 'VAL']
ABPLE_triplets = [''.join(tup) for tup in product('ABPLE', repeat=3)]
relpos = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'same_chain', 'diff_chain']
ABPLE_cols = [str(i) + '_' + ac for i, ac in 
              product(range(1, 11), ABPLE_triplets)]
seqdist_cols = [str(i) + '_' + str(i + 1) + '_' + rp for i, rp in 
                product(range(1, 10), relpos)]
cg_resnames = {'ccn' : ['LYS'], 'gn' : ['ARG'], 'coo' : ['ASP', 'GLU']}

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

def get_atomgroup(environment, cg='ccn', prev_struct=None):
    if cg == 'ccn':
        align_atoms = {'LYS' : ['CD', 'CE', 'NZ']}
    if cg == 'gn':
        align_atoms = {'ARG' : ['NH1', 'CZ', 'NH2']}
    if cg == 'coo':
        align_atoms = {'ASP' : ['OD1', 'CG', 'OD2'], 
                       'GLU' : ['OE1', 'CD', 'OE2']}
    pdb_dir = '/wynton/group/degradolab/nr_pdb/clean_final_pdb/'
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
    selstrs = [selstr_template.format(*scr) for scr in scrs]
    struct = whole_struct.select(
        'same residue as within 5 of ({})'.format(' or '.join(selstrs))
    ).toAtomGroup()
    resnames = []
    for scr, selstr in zip(scrs, selstrs):
        try:
            struct.select(selstr).setOccupancies(2.0)
        except:
            print('Bad SCR: ', biounit, scr)
            return None, None, None
        resnames.append(struct.select(selstr).getResnames()[0])
    try:
        align_coords = np.vstack(
            [struct.select(
                selstrs[0] + ' and name {}'.format(align_atom)
             ).getCoords() for align_atom in align_atoms[resnames[0]]]
        )
    except:
        print('Bad environment to align: ', environment)
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

if __name__ == "__main__":
    cg = sys.argv[1]
    fingerprints_dir = \
        '/wynton/group/degradolab/nr_pdb/{}_fingerprints/'.format(cg)
    hierarchy_dir = \
        '/wynton/group/degradolab/nr_pdb/{}_hierarchy/'.format(cg)
    os.makedirs(hierarchy_dir, exist_ok=True)
    with open(fingerprints_dir + 'fingerprint_cols.txt', 'r') as f:
        fingerprint_cols = np.array(f.read().split(', '))
    for subdir in os.listdir(fingerprints_dir):
        if '.txt' in subdir:
            continue
        for file in os.listdir(os.path.join(fingerprints_dir, subdir)):
            if file.endswith('_fingerprints.npy'):
                fingerprint_array = np.load(
                    os.path.join(fingerprints_dir, subdir, file)
                )
                with open(
                        os.path.join(
                            fingerprints_dir, 
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
                                get_atomgroup(
                                    environment, prev_struct=whole_struct, 
                                    cg=cg
                                )
                        else:
                            atomgroup, resnames, whole_struct = \
                                get_atomgroup(environment, cg=cg)
                            prev_pdb = environment[0][0]
                        if atomgroup is None or \
                                resnames[0] not in cg_resnames[cg]:
                            continue
                        features = fingerprint_cols[fingerprint]
                        features_no_contact = \
                            [feature for feature in features 
                             if np.all([rn not in feature 
                                        for rn in cg_resnames[cg]])]
                        current_res = 1
                        dirs = [resnames[current_res]]
                        while True:
                            if dirs[-1] in aas:
                                ABPLE = [feature for feature in 
                                         features_no_contact 
                                         if feature in ABPLE_cols and 
                                         feature[0] == str(current_res)]
                                if len(ABPLE):
                                    dirs.append(ABPLE[0])
                                else:
                                    break
                            elif dirs[-1] in ABPLE_cols:
                                # print([feature for feature in features_no_contact 
                                #       if feature in seqdist_cols])
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
                        hierarchy_path = '/'.join([hierarchy_dir[:-1]] + dirs)
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

                            


                        
                        
