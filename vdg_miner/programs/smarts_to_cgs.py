import os
import sys
import time
import glob
import pickle
import argparse
from openbabel import openbabel as ob

from vdg_miner.vdg.cg import find_cg_matches

def parse_args():
    parser = argparse.ArgumentParser(
        description="Determine CGs matching a SMARTS pattern."
    )
    parser.add_argument('-s', '--smarts', type=str, required=True, 
                        help="SMARTS pattern.")
    parser.add_argument('-c', '--cg', type=str, 
                        help="The common name for the chemical group. Defaults to the "
                        "SMARTS pattern.")
    parser.add_argument('-p', "--pdb-dir", type=str, required=True,
                        help=("Path to directory containing PDB files "
                              "organized in subdirectories by the "
                              "middle two characters of the PDB ID."))
    parser.add_argument('-o', "--out-dir", type=str, required=True,
                        help="Output directory path.")
    parser.add_argument('-l', "--logfile", default="log", 
                        help="Path to log file.")
    parser.add_argument('-t', "--trial-run", type=int, 
                        help="Number of PDBs to process in a trial run "
                        "used to determine if the script can run to "
                        "completion without errors.")
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    if not args.cg:
        cg = args.smarts
    else:
        cg = args.cg
    matches = {}
    logfile = args.logfile
    num_pdbs_for_trial_run = args.trial_run
    out_dir = os.path.join(args.out_dir, cg)

    print(f'Logfile path: {logfile}')
    with open(logfile, 'a') as file:
        file.write(f"{'='*20} Starting new smarts_to_cgs.py run {'='*20} \n")

    # Reformat and set up outdir
    out_dir = set_up_outdir(num_pdbs_for_trial_run, out_dir, logfile) 

    # Determine which PDBs to process
    all_pdb_paths = sorted(glob.glob(os.path.join(args.pdb_dir, '*', '*.pdb')))
    if num_pdbs_for_trial_run:
        all_pdb_paths = all_pdb_paths[:num_pdbs_for_trial_run]
        with open(logfile, 'a') as file:
            file.write(f'\tExecuting a trial run on the first {num_pdbs_for_trial_run} '
                       'PDBs...\n')
    else:
        with open(logfile, 'a') as file:
            file.write(f'\tProcessing {len(all_pdb_paths)} PDBs...\n')
    
    # Iterate over specified PDBs
    tmpdir = os.path.join(out_dir, 'tmp')
    for pdb_path in all_pdb_paths:
        cg_match_dict, match_mol_objs = find_cg_matches(args.smarts, pdb_path, 
                                                        return_mol_objs=True)
        for ligname, mol_obj in match_mol_objs.items():
            # Write each ligand out as a smiles file, and then use the obabel
            # command-line program to write it out as a 2D sdf file.
            write_out_sdf(mol_obj, ligname, logfile, tmpdir)
        
        matches.update(cg_match_dict)

    # Merge the individual ligand sdf files into a multi-molecule sdf file and then
    # clean up the individual sdf files.
    merged_sdf_name = f'{cg}_ligands.sdf'
    merged_sdf_path = os.path.join(out_dir, merged_sdf_name)
    with open(merged_sdf_path, 'w') as outF:
        for sdf_file in os.listdir(tmpdir):
            if not sdf_file.endswith('.sdf'):
                continue
            with open(os.path.join(tmpdir, sdf_file), 'r') as inF:
                for line in inF:
                    outF.write(line)
    for _file in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, _file))
    os.rmdir(tmpdir)

    # Write out matches as a pkl file
    with open(os.path.join(out_dir, f'{args.cg}_matches.pkl'), 
              'wb') as f:
        pickle.dump(matches, f)
    n_matches = sum([len(v) for v in matches.values()])
    
    # Print out time elapsed and final results
    seconds = time.time() - start_time
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds = round(seconds, 2)
    
    num_structs = len(set([y[0] for y in matches.keys()]))
    with open(logfile, 'a') as file:
        file.write(f'\tNumber of structs (useful for determining the upper limit of ')
        file.write(f'the -n parameter in the downstream generate_fingerprints.py step) ')
        file.write(f': {num_structs}. \n')
        file.write(f'\tFound {n_matches} matches. \n')
        file.write(f"{'='*5} Completed smarts_to_cg.py in {hours} h, ")
        file.write(f"{minutes} mins, and {seconds} secs {'='*5} \n")  

def set_up_outdir(num_pdbs_for_trial_run, out_dir, logfile):
    # Set up output directory
    if num_pdbs_for_trial_run:
        out_dir = out_dir.rstrip('/') + '_trial'
    else:
        out_dir = out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        # Check to see if output directory is empty
        if os.listdir(out_dir):
            with open(logfile, 'a') as file:
                file.write(
                  f'\tThe output directory {out_dir} is not empty. Please remove its '
                  'contents or specify a different output directory path to avoid '
                  'accidental overwriting.\n')
            sys.exit(1)
    return out_dir


def write_out_sdf(mol_obj, ligname, logfile, tmpdir):
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    mol_obj.SetTitle(ligname)
    ob_conversion_to_smiles = ob.OBConversion()
    ob_conversion_to_smiles.SetOutFormat("smiles")
    ob_conversion_to_smiles.WriteString(mol_obj)
    smi_path = os.path.join(tmpdir, f'{ligname}.smi')
    sdf_path = os.path.join(tmpdir, f'{ligname}.sdf')
    ob_conversion_to_smiles.WriteFile(mol_obj, smi_path)
    os.system(f'obabel {smi_path} -O {sdf_path} --gen2D >> {logfile} 2>&1')


if __name__ == '__main__':
    main()