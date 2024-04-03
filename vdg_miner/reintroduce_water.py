import os
import argparse
import prody as pr

from main import get_bio

def biounit_with_hydrogen(biounits, ent_gz_dir, pdb_dir):
    """Given a list of biounits, generate each biounit from the corresponding 
       ent.gz  with ProDy and then add hydrogen with Schrodinger's protassign.

    Parameters
    ----------
    biounits : list
        List of biounits to generate, formatted as a four-letter accession 
        code, followed by '_biounit_', followed by the biounit number.
    ent_gz_dir : str
        Path to the directory at which ent.gz files are stored. Must have a 
        name consiting of two characters: the central two characters of all 
        pdb accession codes of structures in the directory, which themselves 
        should have names formatted as pdb<accession code>.ent.gz
    pdb_dir : str
        Path to the directory at which the biounits with hydrogen added will 
        be output in a subdirectory with the same name as the ent_gz_dir
    """
    for biounit in biounits:
        middle_two = biounit[1:3].lower()
        ent_gz_filename = 'pdb' + biounit[:4].lower() + '.ent.gz'
        ent_gz_path = os.path.join(ent_gz_dir, ent_gz_filename)
        bio = get_bio(ent_gz_path)
        if type(bio) is list:
            biounit_num = int(biounit.split('_')[-1])
            bio = bio[biounit_num]
        water_sel = '(not water) or (water within 4 of protein)'
        bio = bio.select(water_sel).toAtomGroup()
        # write biounit and add hydrogens
        pdb_subdir = os.path.join(pdb_dir, middle_two)
        os.makedirs(pdb_subdir, exist_ok=True)
        pdb_path = os.path.join(pdb_subdir, biounit + '.pdb')
        pr.writePDB(pdb_path, bio)
        # format command for protassign
        cmd = ' '.join(['$SCHRODINGER/utilities/prepwizard', 
                        pdb_path, pdb_path, '-rehtreat', '-nobondorders', 
                        '-nometaltreat', '-samplewater', '-noimpref', 
                        '-use_PDB_pH'])
        os.system(cmd)
        cmd = ' '.join(['mv', biounit + '.log', pdb_subdir])
        os.system(cmd) # move logfile to subdir


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('-e', '--ent-gz-dir', help="Path to directory "
                      "containing ent.gz files from which to generate input "
                      "files for COMBS database generation.")
    argp.add_argument('-p', '--pdb-dir', help="Path to directory "
                      "at which to output biounit structures with water.")
    argp.add_argument('-c', '--cluster-reps', help="File containing the "
                      "biounits and chains that are the lowest-Molprobity "
                      "score representatives of the RCSB sequence clusters "
                      "at 30 percent homology.")
    return argp.parse_args()

if __name__ == "__main__":
    args = parse_args()
    middle_two = os.path.basename(args.ent_gz_dir)
    assert len(middle_two) == 2, 'ent-gz-dir must have a two-character name'
    with open(args.cluster_reps, 'r') as f:
        biounits = list(set(sum([[val.split('/')[-1][:-2] 
                                  for val in line.strip().split()] 
                                 for line in f.readlines()], [])))
    biounits = [b for b in biounits if b[1:3].lower() == middle_two]
    print(biounits)
    biounit_with_hydrogen(biounits, args.ent_gz_dir, args.pdb_dir)
    
    
    