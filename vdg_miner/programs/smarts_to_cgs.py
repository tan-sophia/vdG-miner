import os
import time
import glob
import pickle
import argparse

from vdg_miner.vdg.cg import find_cg_matches

def parse_args():
    parser = argparse.ArgumentParser(
        description="Determine CGs matching a SMARTS pattern."
    )
    parser.add_argument('-s', '--smarts', type=str, required=True, 
                        help="SMARTS pattern.")
    parser.add_argument('-p', "--pdb-dir", type=str, required=True,
                        help=("Path to directory containing PDB files "
                              "organized in subdirectories by the "
                              "middle two characters of the PDB ID."))
    parser.add_argument('-o', "--out-dir", type=str, required=True,
                        help="Output file name, with the pkl extension.")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    matches = {}
    for pdb_path in glob.glob(os.path.join(args.pdb_dir, '*', '*.pdb')):
        matches.update(find_cg_matches(args.smarts, pdb_path))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, args.smarts + '_matches.pkl'), 
              'wb') as f:
        pickle.dump(matches, f)
    time_elapsed = time.time() - start_time
    n_matches = sum([len(v) for v in matches.values()])
    print(f'Found {n_matches} matches in {time_elapsed} seconds.')

if __name__ == '__main__':
    main()