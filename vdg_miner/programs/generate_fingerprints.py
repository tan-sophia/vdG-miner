import os
import sys
import pickle
import argparse
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../vdg'))
from vdg import VDG

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
    parser.add_argument('-n', '--num-jobs', type=int, default=4, 
                        help="Number of jobs, relevant for multi-job "
                             "HPC runs (Default: 4).")
    parser.add_argument('--fingerprints-dir', type=str, default='',
                        help="If set, write fingerprints into this directory "
                             "instead of OUTDIR/fingerprints.")

    return parser.parse_args()

def main():
    args = parse_args()
    logfile = args.logfile
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
    if args.cg_match_dict_pkl is not None:
        structs = set(sorted([key[0] for key in cg_match_dict.keys()]))

        subdicts = [{key: val
                    for key, val in sorted(cg_match_dict.items())
                    if key[0] == struct}
                    for i, struct in enumerate(sorted(structs))
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

if __name__ == '__main__':
    main()
