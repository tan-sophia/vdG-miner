import os
import time
import argparse
import prody as pr

from main import get_bio

def biounit_with_hydrogen(biounits, pdb_gz_dir, pdb_dir):
    """Given a list of biounits, generate each biounit from the corresponding 
       pdb.gz with ProDy and then add hydrogen with Schrodinger's protassign.

    Parameters
    ----------
    biounits : list
        List of biounits to generate, formatted as a four-letter accession 
        code, followed by '_biounit_', followed by the biounit number.
    pdb_gz_dir : str
        Path to the directory at which pdb.gz files are stored. Must have a 
        name consiting of two characters: the central two characters of all 
        pdb accession codes of structures in the directory, which themselves 
        should have names formatted as <ACCESSION CODE>.pdb.gz
    pdb_dir : str
        Path to the directory at which the biounits with hydrogen added will 
        be output in a subdirectory with the same name as the pdb_gz_dir
    """
    orig_cwd = os.getcwd()
    for biounit in biounits:
        try:
            # generate directory for file output
            middle_two = biounit[1:3].lower()
            pdb_subdir = os.path.join(pdb_dir, middle_two)
            os.makedirs(pdb_subdir, exist_ok=True)
            os.chdir(pdb_subdir)
            pdb_path = os.path.join(pdb_subdir, biounit + '.pdb')
            pdb_path_noH = os.path.join(pdb_subdir, biounit + '_noH.pdb')
            log_path = os.path.join(pdb_subdir, biounit + '_noH.log')
            if os.path.exists(pdb_path):
                continue
            # read pdb.gz file and create biounit in ProDy
            pdb_gz_filename = biounit[:4].upper() + '.pdb.gz'
            pdb_gz_path = os.path.join(pdb_gz_dir, pdb_gz_filename)
            bio = get_bio(pdb_gz_path)
            if type(bio) is list:
                biounit_num = int(biounit.split('_')[-1])
                bio = bio[biounit_num]
            # write biounit and add hydrogens
            pr.writePDB(pdb_path_noH, bio)
            cmd = ' '.join(['$SCHRODINGER/utilities/prepwizard', 
                            pdb_path_noH, pdb_path, '-rehtreat', 
                            '-nometaltreat', '-samplewater', '-keepfarwat', 
                            '-use_PDB_pH'])
            os.system(cmd)
            last_mod_time = 0
            while True:
                time.sleep(15)
                current_mod_time = os.path.getmtime(log_path)
                if current_mod_time == last_mod_time:
                    break
                else:
                    last_mod_time = current_mod_time
        except:
            pass
    os.chdir(orig_cwd)


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('-p', '--pdb-gz-dir', help="Path to directory "
                      "containing pdb.gz files from which to generate input "
                      "files for COMBS database generation.")
    argp.add_argument('-f', '--final-pdb-dir', help="Path to directory "
                      "at which to output biounit structures with water.")
    argp.add_argument('-c', '--cluster-reps', help="File containing the "
                      "biounits and chains that are the lowest-Molprobity "
                      "score representatives of the RCSB sequence clusters "
                      "at 30 percent homology.")
    return argp.parse_args()

if __name__ == "__main__":
    args = parse_args()
    middle_two = os.path.basename(args.pdb_gz_dir)
    assert len(middle_two) == 2, 'pdb-gz-dir must have a two-character name'
    with open(args.cluster_reps, 'r') as f:
        biounits = list(set(sum([[val.split('/')[-1][:-2] 
                                  for val in line.strip().split()] 
                                 for line in f.readlines()], [])))
    biounits = [b for b in biounits if b[1:3].lower() == middle_two]
    print(biounits)
    biounit_with_hydrogen(biounits, args.pdb_gz_dir, args.final_pdb_dir)
    
    
    
