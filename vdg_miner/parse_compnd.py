import os
import sys
import gzip
import glob
import pickle

def parse_compnd(ent_gz_path):
    """Parse the COMPND lines of a gzipped PDB file.
    
    Parameters
    ----------
    ent_gz_path : str
        Path to the gzipped PDB file.

    Returns
    -------
    identifier_chains_dict : dict
        Dict pairing polymer entity IDs with lists of chain IDs.
    """
    identifier_chains_dict = {}
    with gzip.open(ent_gz_path, 'rt', encoding='utf-8') as f:
        compnd_begun = False
        while True:
            line = f.readline().strip()
            if line[:6] == 'COMPND':
                compnd_begun = True
                if 'MOL_ID: ' in line:
                    idx = line.index('MOL_ID: ') + 8
                    current_id = int(line[idx:].replace(';',''))
                    identifier_chains_dict[current_id] = []
                if 'CHAIN: ' in line:
                    idx = line.index('CHAIN: ') + 7
                    chains = line[idx:].replace(';','').split(', ')
                    identifier_chains_dict[current_id] += chains
            elif compnd_begun and line[:6] != 'COMPND':
                break
    return identifier_chains_dict

if __name__ == "__main__":
    assert len(sys.argv) == 3, 'PDB directory and output path required'
    identifier_chains_dicts = {}
    prev_middle_two = ''
    counter = 0
    for ent_gz_path in glob.glob(sys.argv[1] + '/*/*.ent.gz'):
        pdb_id = os.path.basename(ent_gz_path)[3:7].upper()
        middle_two = pdb_id[1:3]
        if middle_two != prev_middle_two:
            print(counter)
            counter += 1
            prev_middle_two = middle_two
        identifier_chains_dicts[pdb_id] = parse_compnd(ent_gz_path)
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(identifier_chains_dicts, f)
    