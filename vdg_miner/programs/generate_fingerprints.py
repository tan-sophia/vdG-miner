import os
import time
import pickle
import argparse
import numpy as np
from vdg_miner.vdg.vdg import VDG

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
        file.write(f"{'='*15} Starting new generate_fingerprints.py run {'='*15} \n")

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
        os.path.join(args.outdir, cg, 'fingerprints')
    os.makedirs(fingerprints_dir, exist_ok=True)

    all_fingerprint_labels, all_environments = [], []
    if args.cluster_reps_file is not None:
        with open(args.cluster_reps_file, 'r') as f:
            lines = [[string.split('/')[-1] 
                      for string in line.strip().split()] 
                     for i, line in enumerate(f.readlines())
                     if i % args.num_jobs == args.job_index]
        for i, line in enumerate(lines):
            fingerprint_labels, environments = \
                vdg.mine_pdb(chain_cluster=line)
            if not len(fingerprint_labels) or not len(environments):
                continue
            all_fingerprint_labels.append(fingerprint_labels)
            all_environments.append(environments)
    elif args.cg_match_dict_pkl is not None:
        structs = set([key[0] for key in cg_match_dict.keys()])
        subdicts = [{key: val for key, val in cg_match_dict.items() 
                     if key[0] == struct} for i, struct in enumerate(structs)
                     if i % args.num_jobs == args.job_index]
        for i, subdict in enumerate(subdicts):
            fingerprint_labels, environments = \
                vdg.mine_pdb(cg_match_dict=subdict)
            if not len(fingerprint_labels) or not len(environments):
                continue
            all_fingerprint_labels.append(fingerprint_labels)
            all_environments.append(environments)
    else:
        raise ValueError("Either cluster-reps-file or cg-match-dict "
                         "must be provided.")
    with open(os.path.join(fingerprints_dir, 'fingerprint_cols.txt'), 
              'w') as f:
        f.write(', '.join(vdg.fingerprint_cols))
    for fingerprint_labels, environments in \
            zip(all_fingerprint_labels, all_environments):
        middle_two = environments[0][0][0][1:3].lower()
        chain = '_'.join(environments[0][0][:3])
        os.makedirs(os.path.join(fingerprints_dir, middle_two), exist_ok=True)
        env_outpath = os.path.join(fingerprints_dir, 
                                   middle_two, 
                                   chain + '_environments.txt')
        fp_outpath = os.path.join(fingerprints_dir, 
                                  middle_two, 
                                  chain + '_fingerprints.npy')
        # generate fingerprint array from fingerprint_labels
        fingerprints = np.zeros((len(fingerprint_labels), 
                                 len(vdg.fingerprint_cols)), dtype=bool)
        for i, labels in enumerate(fingerprint_labels):
            for label in labels:
                fingerprints[i, vdg.fingerprint_cols.index(label)] = True
        # save environments and fingerprints
        with open(env_outpath, 'w') as f:
            for env in environments:
                f.write(repr(env) + '\n')
        np.save(fp_outpath, fingerprints)
    
    
    # Print out time elapsed
    seconds = time.time() - start_time
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds = round(seconds, 2)
    
    
    with open(logfile, 'a') as file:
        file.write(f"{'='*2} Completed generate_fingerprints.py in {hours} h, ")
        file.write(f"{minutes} mins, and {seconds} secs {'='*2} \n")

if __name__ == '__main__':
    main()