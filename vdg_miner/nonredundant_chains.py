import sys
import glob
import pickle

from tqdm import tqdm

def cluster_rep(cluster, entity_ids, scores_dict, pdb_dir):
    """Find the cluster member with the lowest Molprobity score.
    
    Parameters
    ----------
    cluster : list
        List of entities within a sequence cluster, expressed as 
        4-letter PDB accession codes followed by an underscore and 
        an integer entity ID.
    entity_ids : dict
        Dictionary that has as keys the 4-letter codes of every 
        PDB structure and as values dictionaries that themselves 
        have as keys the integer entity IDs therein and as values 
        the lists of chain IDs that pertain to those entity IDs.
    scores_dict : dict
        A dictionary that has as keys the biounits in a database 
        and as values the Molprobity scores of those biounits.
    pdb_dir : str
        Path to the directory in which the reduced PDB files for 
        which Molprobity has been run are located (in subdirectories 
        organized by the second to characters of the PDB accession 
        codes).
    """
    min_score = 2.
    min_score_pdbs = []
    for mem in cluster:
        if mem[4] != '_':
            continue
        pdb_id, entity_id = mem.split('_')
        entity_id = int(entity_id)
        try:
            chains = entity_ids[pdb_id][entity_id]
        except: 
            continue
        middle_two = pdb_id[1:3].lower()
        glob_str = pdb_dir + '/' + middle_two + '/' + pdb_id + '*out'
        for outfile in glob.glob(glob_str):
            prefix = outfile[:-15]
            pdb_file = prefix + '.pdb'
            score = scores_dict[outfile]
            if score == min_score:
                with open(pdb_file, 'r') as f:
                    pdb_chains = set([line[21] for line in f.readlines() 
                                      if 'ATOM' in line])
                for chain in chains:
                    if chain in pdb_chains:
                        min_score_pdbs.append(prefix + '_' + chain)
            if score < min_score:
                min_score = score
                min_score_pdbs = []
                with open(pdb_file, 'r') as f:
                    pdb_chains = set([line[21] for line in f.readlines() 
                                      if 'ATOM' in line])
                for chain in chains:
                    if chain in pdb_chains:
                        min_score_pdbs.append(prefix + '_' + chain)
    if len(min_score_pdbs):
        return ' '.join(min_score_pdbs)
    else:
        return ''


if __name__ == "__main__":
    assert len(sys.argv) == 2, 'Path to reduced PDB directory required.'
    with open("clusters-by-entity-30.txt", "r") as f:
        clusters = [line.strip().split() for line in f.readlines()]
    with open("entity_ids.pkl", "rb") as f:
        entity_ids = pickle.load(f)
    with open("scores_dict.pkl", "rb") as f:
        scores_dict = pickle.load(f)
    cluster_reps = []
    for cluster in tqdm(clusters):
        rep = cluster_rep(cluster, entity_ids, 
                          scores_dict, sys.argv[1])
        if len(rep):
            cluster_reps.append(rep)
    with open("cluster_reps.txt", "w") as f:
        for rep in cluster_reps:
            f.write(rep + '\n')
    