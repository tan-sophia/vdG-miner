import os
import argparse

_probe = '/wynton/home/degradolab/rkormos/probe/probe'

def parse_args():
    argp = argparse.ArgumentParser('Run probe on a set of biounits.')
    argp.add_argument('-c', '--cluster-reps', 
                      help='File containing the biounits and chains that are '
                           'the lowest-Molprobity score representatives of '
                           'the RCSB sequence clusters at 30 percent '
                           'homology.')
    argp.add_argument('-p', '--pdb-dir',
                      help='Path to the directory in which the reduced PDB '
                           'files on which probe should be run are located. '
                           'This directory should have a two-letter name, '
                           'which is the second and third characters of the '
                           'PDB accession codes of the structures in the '
                           'directory.')
    argp.add_argument('-o', '--out-dir',
                      help='Path to the directory in which the output probe '
                           'files are to be written. This directory will be '
                           'organized into subdirectories by the second and '
                           'third characters of the PDB accession codes of '
                           'the biounits.')
    return argp.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.cluster_reps, 'r') as f:
        cluster_reps = [line.strip().split() for line in f.readlines()]
    middle_two = os.path.basename(args.pdb_dir)
    for cluster_rep_list in cluster_reps:
        for cluster_rep in cluster_rep_list:
            split = cluster_rep.split('/')
            if split[1] != middle_two:
                continue
            biounit = split[-1][:-2]
            chain = split[-1][-1]
            pdb_file = os.path.join(args.pdb_dir, biounit + '.pdb')
            out_file = os.path.join(args.out_dir, middle_two, 
                                    split[-1] + '.probe')
            cmd = ('{} -U -SEGID -CON -NOFACE -Explicit -WEAKH -DE32 -4 -ON '
                   '"WATER,CHAIN_{}" "ALL" {} > {}').format(_probe, chain, 
                                                            pdb_file, out_file)
            os.system(cmd)
            if os.path.exists(out_file):
                gzip_cmd = 'gzip {}'.format(out_file)
                os.system(gzip_cmd)
            