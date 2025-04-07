import os
import sys
import time
import pickle
import argparse
import numpy as np
from numba import njit, prange, set_num_threads
sys.path.append(os.path.join(os.path.dirname(__file__), '../vdg'))
from vdg import VDG

set_num_threads(10) 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fingerprints for a set of PDB files."
    )
    parser.add_argument('-c', '--cg', type=str, required=True, 
                        help="Chemical group name.")
    parser.add_argument('-r', '--cluster-reps-file', type=str, 
                        help="File containing the biounits and chains "
                             "that are the lowest-Molprobity score "
                             "representatives of the RCSB sequence clusters "
                             "at 30 percent homology. Either this or "
                             "cg-match-dict must be provided.")
    parser.add_argument('-m', '--cg-match-dict-pkl', type=str, 
                        help="Path to the pickled CG match dictionary if "
                             "the CG is not proteinaceous. Either this or "
                             "cluster-reps-file must be provided.")
    parser.add_argument('-p', '--pdb-dir', required=True, type=str, 
                        help="Path to the directory in which the reduced "
                             "PDB files containing the structures to mine "
                             "are located. This directory should have a "
                             "two-letter name, which is the second and "
                             "third characters of the PDB accession codes "
                             "of the structures in the directory.")
    parser.add_argument('-b', '--probe-dir', required=True, type=str, 
                        help="Path to the directory in which the gzipped "
                             "probe files for the structures to mine "
                             "are located. This directory should contain "
                             "subdirectories with two-letter names, which "
                             "are the second and third characters of the "
                             "PDB accession codes of the structures with "
                             "probe files in the directory.")
    parser.add_argument('-v', '--validation-dir', type=str, default='',
                        help="Path to the directory in which the gzipped "
                             "validation reports for the structures to mine "
                             "are located. This directory should contain "
                             "subdirectories with two-letter names, which "
                             "are the second and third characters of the "
                             "PDB accession codes of the structures with "
                             "validation reports in the directory.")
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="Output directory wherein a new directory for "
                             "output files will be created.")
    parser.add_argument('-l', "--logfile", default="log", 
                        help="Path to log file.")
    parser.add_argument('-j', '--job-index', type=int, default=0, 
                        help="Index for the current job, relevant for "
                             "multi-job HPC runs (Default: 0).")
    parser.add_argument('-n', '--num-jobs', type=int, default=1, 
                        help="Number of jobs, relevant for multi-job "
                             "HPC runs (Default: 1).")
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    logfile = args.logfile
    with open(logfile, 'a') as file:
        file.write(f"{'='*20} Starting generate_fingerprints.py run {'='*20} \n")

    cg = args.cg
    if args.cg_match_dict_pkl is not None:
        with open(args.cg_match_dict_pkl, 'rb') as f:
            cg_match_dict = pickle.load(f)
        cg_natoms = len(cg_match_dict[list(cg_match_dict.keys())[0]])
        vdg = VDG(cg, pdb_dir=args.pdb_dir, probe_dir=args.probe_dir,
                  validation_dir=args.validation_dir, cg_natoms=cg_natoms)
    else:
        vdg = VDG(cg, pdb_dir=args.pdb_dir, probe_dir=args.probe_dir,
                  validation_dir=args.validation_dir)
    fingerprints_dir = \
        os.path.join(args.outdir, 'fingerprints')
    os.makedirs(fingerprints_dir, exist_ok=True)

    all_fingerprints, all_environments = [], []
    ''' The below is only used for design, not docking.
    if args.cluster_reps_file is not None:
        with open(args.cluster_reps_file, 'r') as f:
            lines = [[string.split('/')[-1] 
                      for string in line.strip().split()] 
                     for i, line in enumerate(f.readlines())
                     if i % args.num_jobs == args.job_index]
        for i, line in enumerate(lines):
            fingerprints, environments = \
                vdg.mine_pdb(chain_cluster=line)
            if not len(fingerprints) or not len(environments):
                continue
            all_fingerprints.append(fingerprints)
            all_environments.append(environments)
    '''
    if args.cg_match_dict_pkl is not None:
        structs = set([key[0] for key in cg_match_dict.keys()])
        # If there are too many structures to process (e.g., > 20,000),
        # select a subset of structures with maximum PDB ID diversity
        max_num_structs = 20000
        print('Number of structs:', len(structs))
        if len(structs) > max_num_structs:
            with open(logfile, 'a') as file:
                file.write(f'\tThere are over {max_num_structs} structures in the matches '
                    f'dict ({len(structs)}). Selecting a subset of {max_num_structs} structures with '
                    'maximum PDB ID diversity.\n')
            structs = select_diverse_pdbIDs(list(structs), max_num_structs)
        
        subdicts = [{key: val for key, val in cg_match_dict.items() 
                     if key[0] == struct} for i, struct in enumerate(structs)
                     if i % args.num_jobs == args.job_index]
        for i, subdict in enumerate(subdicts):
            fingerprints, environments = \
                vdg.mine_pdb(logfile=logfile, cg_match_dict=subdict)
            if not len(fingerprints) or not len(environments):
                continue
            all_fingerprints.append(fingerprints)
            all_environments.append(environments)
    else:
        raise ValueError("Either cluster-reps-file or cg-match-dict "
                         "must be provided.")
    with open(os.path.join(fingerprints_dir, 'fingerprint_cols.txt'), 
              'w') as f:
        f.write(', '.join(vdg.fingerprint_cols))
    for fingerprints, environments in \
            zip(all_fingerprints, all_environments):
        middle_two = environments[0][0][0][1:3].lower()
        chain = '_'.join(environments[0][0][:3])
        os.makedirs(os.path.join(fingerprints_dir, middle_two), exist_ok=True)
        env_outpath = os.path.join(fingerprints_dir, 
                                   middle_two, 
                                   chain + '_environments.txt')
        fp_outpath = os.path.join(fingerprints_dir, 
                                  middle_two, 
                                  chain + '_fingerprints.npy')
        # save environments and fingerprints
        with open(env_outpath, 'w') as f:
            for env in environments:
                f.write(repr(env) + '\n')
        np.save(fp_outpath, fingerprints)
    
    
    # Print out time elapsed
    seconds = time.time() - start_time
    hours = round(seconds // 3600)
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds = round(seconds, 2)
    
    
    with open(logfile, 'a') as file:
        file.write(f'\t{len(all_fingerprints)} fingerprints generated.\n')
        file.write(f"Completed generate_fingerprints.py in {hours} h, ")
        file.write(f"{minutes} mins, and {seconds} secs.\n")

def select_diverse_pdbIDs(strings, k): # k = max_num_structs
    ascii_array = strings_to_ascii_array(strings)
    selected_indices = select_diverse_subset_parallel(ascii_array, k)
    return [strings[i] for i in selected_indices]

def strings_to_ascii_array(strings):
    return np.array([[ord(c) for c in s] for s in strings], dtype=np.uint8)

@njit
def hamming(s1, s2):
    dist = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            dist += 1
    return dist

@njit(parallel=True)
def update_min_dists(data, selected_idx, selected_mask, min_dists):
    n = data.shape[0]
    for i in prange(n):
        if selected_mask[i] == 0:
            dist = hamming(data[selected_idx], data[i])
            if dist < min_dists[i]:
                min_dists[i] = dist

@njit
def select_diverse_subset_parallel(data, k):
    n = data.shape[0]
    selected = [0]  # start with first point
    selected_mask = np.zeros(n, dtype=np.uint8)
    selected_mask[0] = 1
    min_dists = np.full(n, 255, dtype=np.uint8)

    # Initial distance pass
    for i in range(1, n):
        min_dists[i] = hamming(data[0], data[i])

    for _ in range(1, k):
        # Select max of min distances
        max_idx = -1
        max_val = -1
        for i in range(n):
            if selected_mask[i] == 0 and min_dists[i] > max_val:
                max_val = min_dists[i]
                max_idx = i

        selected.append(max_idx)
        selected_mask[max_idx] = 1

        # Parallel update of min distances
        update_min_dists(data, max_idx, selected_mask, min_dists)

    return selected


if __name__ == '__main__':
    main()
