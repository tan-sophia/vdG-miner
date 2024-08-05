import os
import sys
import numpy as np
from scipy.spatial.distance import cdist

pdb_dir = sys.argv[1]
subdirs = [os.path.join(pdb_dir, subdir) for subdir in os.listdir(pdb_dir)]
for subdir in subdirs:
    pdb_files = [os.path.join(subdir, pdb_file) for pdb_file in os.listdir(subdir) 
                 if pdb_file.endswith('.pdb')]
    for pdb_file in pdb_files:
        print(pdb_file)
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        # line_to_atom, atom_to_line = [-1] * len(lines), []
        # counter = 0
        x, y, z, is_H, segis = np.empty(0, dtype=float), \
                               np.empty(0, dtype=float), \
                               np.empty(0, dtype=float), \
                               np.empty(0, dtype=bool), \
                               np.empty(0, dtype=int)
        atom_to_line = []
        for i, line in enumerate(lines):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = np.append(x, float(line[30:38]))
                y = np.append(y, float(line[38:46]))
                z = np.append(z, float(line[46:54]))
                is_H = np.append(is_H, line[76:78].strip() == 'H')
                if line[72:76] != '    ':
                    segis = np.append(segis, int(line[72:76]))
                else:
                    segis = np.append(segis, -1)
                atom_to_line.append(i)
        coords = np.vstack((x, y, z)).T
        sqdists = cdist(coords[is_H], coords[~is_H], metric='sqeuclidean')
        neighbors = -np.ones(len(coords), dtype=int)
        neighbors[is_H] = np.where(~is_H)[0][np.argmin(sqdists, axis=1)]
        with open(pdb_file, 'w') as f:
            counter = 0
            for line in lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    if is_H[counter]:
                        assert line[76:78].strip() == 'H'
                        # assert line[72:76] == '    '
                        neighbor_line = atom_to_line[neighbors[counter]]
                        f.write(line[:72] + 
                                lines[neighbor_line][72:76] + 
                                line[76:])
                    else:
                        f.write(line)
                    counter += 1
                else:
                    f.write(line)