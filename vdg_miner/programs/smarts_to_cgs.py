import os
import sys
import subprocess
import time
import glob
import pickle
import argparse
from openbabel import openbabel as ob
sys.path.append(os.path.join(os.path.dirname(__file__), '../vdg'))
from cg import find_cg_matches

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
    out_dir = args.out_dir    

    print(f'\nLogfile path: {logfile}\n')
    # Set up log dir
    log_dir = os.path.dirname(logfile)
    if log_dir != '':
        os.makedirs(log_dir, exist_ok=True)

    with open(logfile, 'a') as file:
        file.write(f"{'='*24} Starting smarts_to_cgs.py run {'='*24} \n")

    # Set up outdir
    out_dir = set_up_outdir(out_dir, logfile) 

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
    if len(all_pdb_paths) == 0:
        raise ValueError('No PDBs in the input dir specified by the -p flag.')
    tmpdir = os.path.join(out_dir, 'tmp')
    num_failed_ligs = 0
    for pdb_path in all_pdb_paths:
        if not os.path.exists(pdb_path):
            with open(logfile, 'a') as file:
                file.write(f'\tPDB {pdb_path} does not exist.\n')
            continue
        cg_match_dict, match_mol_objs = find_cg_matches(args.smarts, pdb_path, 
                                                        return_mol_objs=True)
        for ligname, mol_obj in match_mol_objs.items():
            # Write each ligand out as a smiles file, and then use the obabel
            # command-line program to write it out as a 2D sdf file.
            num_failed_ligs = write_out_sdf(mol_obj, ligname, logfile, tmpdir, 
                                            num_failed_ligs)
        
        matches.update(cg_match_dict)

    # Merge the individual ligand sdf files into a multi-molecule sdf file and then
    # clean up the individual sdf files.
    merged_sdf_name = f'{cg}_ligands.sdf'
    merged_sdf_path = os.path.join(out_dir, merged_sdf_name)
    no_ligs_msg = 'No ligands contain the specified SMARTS pattern.\n'
    if not os.path.exists(tmpdir):
        with open(logfile, 'a') as file:
            file.write(no_ligs_msg)
        raise ValueError(no_ligs_msg)
    if not os.listdir(tmpdir):
        with open(logfile, 'a') as file:
            file.write(no_ligs_msg)
        raise ValueError(no_ligs_msg)
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
    n_unique_ligs = len(set([k[-1] for k in matches.keys()]))
    
    # Print out time elapsed and final results
    seconds = time.time() - start_time
    hours = round(seconds // 3600)
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds = round(seconds, 2)
    
    num_structs = len(set([y[0] for y in matches.keys()]))
    with open(logfile, 'a') as file:
        #file.write(f'\tNumber of structs (useful for determining the upper limit of ')
        #file.write(f'the -n parameter in the downstream generate_fingerprints.py step) ')
        #file.write(f': {num_structs}. \n')
        file.write(f'\t{n_unique_ligs} unique ligs w/ SMARTS found in database.\n')
        file.write(f'\t{n_matches} instances of SMARTS interacting with protein.\n') 
        file.write(f'\t{num_failed_ligs} ligands failed.\n')
        file.write(f"Completed smarts_to_cg.py in {hours} h, ")
        file.write(f"{minutes} mins, and {seconds} secs.\n") 

    # Clean up the log file. obabel outputs a message for each molecule it parses, so
    # remove all the lines corresponding to molecules it successfully parses (so that
    # it's easier to see the error messages).
    sed_command = f"sed -i '/1 molecule converted/d' \"{logfile}\""
    subprocess.run(sed_command, shell=True, check=True)

def set_up_outdir(out_dir, logfile):
    # Set up output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def write_out_sdf(mol_obj, ligname, logfile, tmpdir, num_failed_ligs):
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    mol_obj.SetTitle(ligname)
    ob_conversion_to_smiles = ob.OBConversion()
    ob_conversion_to_smiles.SetOutFormat("smiles")
    ob_conversion_to_smiles.WriteString(mol_obj)
    smi_path = os.path.join(tmpdir, f'{ligname}.smi')
    sdf_path = os.path.join(tmpdir, f'{ligname}.sdf')
    ob_conversion_to_smiles.WriteFile(mol_obj, smi_path)
    #os.system(f'obabel "{smi_path}" -O "{sdf_path}" --gen2D >> "{logfile}" 2>&1')
    
    # The code may get stuck on a ligand. If the subprocess does not complete within
    # a few mins, then kill it and move on.
    try:
        result = subprocess.run(
            ['obabel', smi_path, '-O', sdf_path, '--gen2D'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300)
    except subprocess.TimeoutExpired:
        with open(logfile, 'a') as file:
            file.write(f"\t\tobabel timeout expired for {os.path.basename(smi_path)}\n")
        # Delete the failed smi and sdf files.
        time.sleep(10)
        if os.path.exists(smi_path):
            os.remove(smi_path)
        if os.path.exists(sdf_path):
            os.remove(sdf_path)
        num_failed_ligs += 1
    return num_failed_ligs

if __name__ == '__main__':
    main()

