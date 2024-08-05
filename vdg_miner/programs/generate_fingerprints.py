import os
import sys
import numpy as np
from vdg_miner.vdg.vdg import VDG

with open('/wynton/group/degradolab/nr_pdb/cluster_reps.txt', 'r') as f:
    lines = [[string.split('/')[-1] for string in line.strip().split()] 
             for line in f.readlines()]

cg = sys.argv[1]
vdg = VDG(cg)
fingerprints_dir = \
    '/wynton/group/degradolab/nr_pdb/{}_fingerprints/'.format(cg)
os.makedirs(fingerprints_dir, exist_ok=True)
if len(sys.argv) > 2:
    line_idx = int(sys.argv[2])
else:
    line_idx = 0
if line_idx == 0:
    with open(fingerprints_dir + 'fingerprint_cols.txt', 'w') as f:
        f.write(', '.join(vdg.fingerprint_cols))
fingerprints, environments = vdg.mine_pdb(lines[line_idx])

middle_two = environments[0][0][0][1:3].lower()
chain = '_'.join(environments[0][0][:3])
os.makedirs(os.path.join(fingerprints_dir, middle_two), exist_ok=True)
env_outpath = os.path.join(fingerprints_dir, 
                           middle_two, 
                           chain + '_environments.txt')
fp_outpath = os.path.join(fingerprints_dir, 
                          middle_two, 
                          chain + '_fingerprints.npy')

with open(env_outpath, 'w') as f:
    for env in environments:
        f.write(repr(env) + '\n')
np.save(fp_outpath, fingerprints)
