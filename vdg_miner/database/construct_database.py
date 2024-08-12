import os
import sys
import gzip
import errno
import signal
import shutil
import pickle
import argparse
import traceback

import numpy as np
import prody as pr

from copy import deepcopy
from functools import wraps
from scipy.spatial.distance import cdist

module = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, module)

from database.readxml import extract_global_validation_values
from constants import three_to_one, non_prot_sel

"""
Updated pdb files, validation reports, and sequence clusters should be 
downloaded via the pdb ftp server:

> rsync -rlpt -v -z --delete --port=33444 
  rsync.rcsb.org::ftp_data/structures/divided/pdb/ $LOCAL_PDB_MIRROR_PATH

> rsync -rlpt -v -z --delete --include="*/" --include="*.xml.gz" --exclude="*"  
  --port=33444 rsync.rcsb.org::ftp/validation_reports/ $LOCAL_VALIDATION_PATH

> wget -O $LOCAL_CLUSTERS_PATH
  https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt

These three paths should be provided using the -e, -v, and -c arguments to 
this script, respectively.
"""

class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """Raise a TimeoutError if a function takes longer than the specified time.
    
    Parameters
    ----------
    seconds : int
        Number of seconds after which to raise the TimeoutError.
    error_message : str
        Message to include in the TimeoutError.

    Returns
    -------
    decorator : function
        Decorator function that raises a TimeoutError if the wrapped 
        function takes longer than the specified time.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def read_clusters(cluster_file, job_id, num_jobs):
    """Read the sequence clusters from the RCSB cluster file.

    Parameters
    ----------
    cluster_file : str
        Path to the file containing the RCSB-curated sequence clusters.
    job_id : int
        Index for the current job, relevant for multi-job HPC runs.
    num_jobs : int
        Number of jobs, relevant for multi-job HPC runs.

    Returns
    -------
    clusters : list
        List of lists of members of clusters for the current job, stored 
        as tuples of PDB accession codes and associated polymer entity IDs.
    """
    clusters = []
    with open(cluster_file, 'r') as f:
        for line in f:
            clusters.append([])
            for member in line.split():
                mem_split = member.split('_')
                if len(mem_split[0]) == 4: # exclude non-PDB entries
                    clusters[-1].append(member.split('_'))
            if len(clusters[-1]) == 0:
                clusters.pop()
    # assign to the current job approximately 1 / num_jobs of the clusters
    cluster_cumsum = np.cumsum([len(c) for c in clusters])
    cluster_idxs = np.argwhere(cluster_cumsum % num_jobs == job_id).flatten()
    return [c for i, c in enumerate(clusters) if i in cluster_idxs]


def filter_clusters(clusters, validation_dir, min_res=2.0, max_r=0.3):
    """Filter the clusters to include only those with valid resolution and R.

    Parameters
    ----------
    clusters : list
        List of lists of members of clusters, stored as tuples of PDB 
        accession codes and associated polymer entity IDs.
    validation_dir : str
        Path to directory containing xml.gz files with validation report data.
    min_res : float, optional
        Minimum resolution (in Angstroms) to permit for a PDB structure 
        (Default: 2.0).
    max_r : float, optional
        Maximum R value to permit for a PDB structure (Default: 0.3).

    Returns
    -------
    filtered_clusters : list
        List of lists of members of clusters, stored as tuples of PDB 
        accession codes and associated polymer entity IDs, that have valid 
        resolution and R.
    """
    filtered_clusters = []
    for cluster in clusters:
        filtered_clusters.append([])
        for pdb, entity in cluster:
            middle_two = pdb[1:3].lower()
            pdb_acc = pdb.lower()
            validation_file = os.path.join(validation_dir, 
                                           middle_two, pdb_acc, 
                                           pdb_acc + '_validation.xml.gz')
            resolution, r_obs = \
                extract_global_validation_values(validation_file)
            if resolution is not None and r_obs is not None:
                if resolution <= min_res and r_obs <= max_r:
                    filtered_clusters[-1].append((pdb, entity))
        if len(filtered_clusters[-1]) == 0:
            filtered_clusters.pop()
    return filtered_clusters


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


def get_segs_chains_resnums(atomgroup, selection, resnames=False):
    """Get a set containing tuples of segments, chains, and resnums.

    Parameters
    ----------
    atomgroup : prody.atomic.atomgroup.AtomGroup
        ProDy AtomGroup for a protein structure.
    selection : str
        String specifying subset of atoms in the ProDy selection algebra.
    resnames : bool
        If True, return residue names in each tuple as well.

    Returns
    -------
    segs_chains_resnums : set
        Set containing tuples of segments, chains, and resnums for 
        each residue that matches the selection string.
    """
    sel = atomgroup.select(selection)
    if sel is None:
        return set()
    if resnames:
        return set(zip(sel.getSegnames(), sel.getChids(), sel.getResnums(), 
                       sel.getResnames()))
    else:
        return set(zip(sel.getSegnames(), sel.getChids(), sel.getResnums()))


def get_author_assigned_biounits(pdb_file):
    """Given a gzipped PDB file, obtain the author-assigned biounits.

    Parameters
    ----------
    pdb_file : str
        Path to gzipped PDB file for which to obtain author-assigned biounits.

    Returns
    -------
    biomols : list
        List of integer indices of biological assemblies.
    """
    _str = 'AUTHOR DETERMINED BIOLOGICAL UNIT'
    biomols = []
    with gzip.open(pdb_file, 'rt') as infile:
        for line in infile:
            if line[:10] == 'REMARK 350':
                break
        for line in infile:
            if line[:10] != 'REMARK 350':
                break
            if 'BIOMOLECULE:' in line:
                biomol = int(line.strip().split('BIOMOLECULE:')[-1])
                continue
            if _str in line:
                biomols.append(biomol)
    return biomols


@timeout(5)
def get_bio(path, return_header=False):
    """Given the path to a gzipped PDB file, return its biological assemblies.

    Parameters
    ----------
    path : str
        Path to gzipped PDB file for which to return biological assemblies.\
    return_header : bool, optional
        If True, return the header of the PDB file as well.

    Returns
    -------
    bio : prody.AtomGroup or list
        ProDy AtomGroup or list of ProDy AtomGroups for the biological 
        assemblies of the structure.
    """
    with gzip.open(path, 'rt') as f:
        return pr.parsePDBStream(f, biomol=True, header=return_header)


def write_biounits(ent_gz_path, pdb_outdir, max_ligands=None, 
                   xtal_only=True, write=True):
    """For an ent.gz file, write the biological assemblies as PDB files.

    Parameters
    ----------
    ent_gz_path : str
        Path to the ent.gz file for a PDB structure to convert to a 
        biological assembly.
    pdb_outdir : str
        Directory at which to output PDB files for biological assemblies.
    max_ligands : int, optional
        Maximum number of heteroatom (i.e. non-protein, non-nucleic, and 
        non-water) residues to permit in a biological assembly.
    xtal_only : bool, optional
        Whether to only consider X-ray crystallography structures.
    write : bool, optional
        If False, do not write the biological assemblies to PDB files.

    Returns
    -------
    bio_paths : list
        List of paths (within pdb_outdir) to PDB files containing the 
        author-assigned biological assemblies of the input PDB structure, 
        or all biounits if no author-assigned biounits are denoted.
    """
    bio_paths = []
    try:
        bio, header = get_bio(ent_gz_path, return_header=True)
        if xtal_only and 'X-RAY' not in header['experiment']:
            return []
        pdb_code = ent_gz_path.split('/')[-1][3:7]
        if type(bio) != list:
            bio = [bio]
        bio_list = [int(b.getTitle().split()[-1]) for b in bio]
        author_assigned = get_author_assigned_biounits(ent_gz_path)
        if len(author_assigned) > 0:
            bio = [bio[bio_list.index(i)] for i in author_assigned]
            bio_list = author_assigned
        n_near = [len(
                      get_segs_chains_resnums(
                          b, 
                          non_prot_sel + ' within 4 of protein'
                      )
                  ) for b in bio]
        if type(max_ligands) is int:
            n_ligands = \
                [len(get_segs_chains_resnums(b, 'not water hetero')) 
                    for b in bio]
            bio = [b for b, nl, nn in zip(bio, n_ligands, n_near) 
                    if nl < max_ligands and nn > 0]
        else:
            bio = [b for b, nn in zip(bio, n_near) if nn > 0] 
        for i, b in enumerate(bio):
            if not b.select('protein'):
                continue
            b = b.select('protein or same residue as within 4 of protein')

            # determine one-letter sequence
            resnames = [res.getResname() for res in bio[i].iterResidues()]
            seq = ''.join([three_to_one[rn] if rn in three_to_one.keys() 
                            else 'X' for rn in resnames])
            # write biounit to PDB file
            bio_dirname = os.path.join(pdb_outdir, pdb_code[1:3], pdb_code)
            os.makedirs(bio_dirname, exist_ok=True)
            bio_path = os.path.join(bio_dirname, 
                                    pdb_code.upper() + '_biounit_' + 
                                    str(bio_list[i]) + '.pdb')
            if write and not os.path.exists(bio_path):
                pr.writePDB(bio_path, bio[i])
            bio_paths.append(bio_path)
    except Exception:
        print('**************************************************')
        traceback.print_exc(file=sys.stdout)
    return bio_paths


def prep_biounits(biounit_paths, prepwizard_path):
    """Add hydrogen to biological assemblies with Schrodinger Prepwizard.

    Parameters
    ----------
    biounit_oaths : list
        List of paths to PDB files of biological assemblies to which to add
        hydrogen with Schrodinger Prepwizard.
    prepwizard_path : str
        Path to Schrodinger Prepwizard binary.
    """
    cwd = os.getcwd()
    for biounit_path in biounit_paths:
        os.chdir(os.path.dirname(biounit_path))
        if os.path.exists(biounit_path[:-4] + '.log'):
            continue # skip if already processed
        # format and execute command for Prepwizard
        cmd = ' '.join([prepwizard_path, biounit_path, biounit_path, 
                        '-rehtreat', '-nobondorders', '-samplewater', 
                        '-noimpref', '-use_PDB_pH'])
        os.system(cmd)
    os.chdir(cwd)


def run_molprobity(pdb_path, molprobity_path, cutoff=2.0):
    """Run Molprobity on a PDB file and remove the PDB if score > cutoff.

    Parameters
    ----------
    pdb_path : list
        The path to the PDB file to be scored by MolProbity.
    molprobity_path : str
        Path to Molprobity binary.
    cutoff : float
        The MolProbity score cutoff for a PDB to be considered good.

    Returns
    -------
    score : float
        The MolProbity score of the PDB file.
    """
    pdb_dirname = os.path.dirname(pdb_path)
    parent_dirname = os.path.dirname(pdb_dirname)
    out_path = pdb_path[:-4] + '_molprobity.out'
    cmd = [molprobity_path, pdb_dirname, '>', out_path]
    os.system(' '.join(cmd))
    if os.path.exists(out_path):
        return
    with open(out_path, 'rb') as f:
        # read last line of file to get MolProbity score
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        score = float(f.readline().decode().split(':')[-2])
    if score <= cutoff:
        cmd = ['mv', pdb_path, parent_dirname, ';' 
               'mv', out_path, parent_dirname, ';', 
               'rm', '-r', pdb_dirname]
    else:
        cmd = ['rm', '-r', pdb_dirname]
    os.system(' '.join(cmd))
    return score


def ent_gz_dir_to_vdg_db_files(ent_gz_dir, pdb_outdir, 
                               final_cluster_outpath, clusters, 
                               molprobity_path, prepwizard_path, 
                               max_ligands=25, retry=False):
    """Generate vdG database files from ent.gz files and validation reports.

    Parameters
    ----------
    ent_gz_dir : str
        Path to directory containing ent.gz files from which to generate input 
        files for COMBS database generation.
    pdb_outdir : str
        Directory at which to output fully prepared PDB files for biological 
        assemblies with minimal MolProbity scores.
    final_cluster_outpath : str
        Path to output file at which to specify the biounits and chains that
        are the lowest-Molprobity score representatives of the RCSB sequence
        clusters.
    clusters : list
        List of list of clusters (tuples of PDB accession codes and entity IDs) 
        to prepare with prepwizard and assess with MolProbity for database 
        membership.
    molprobity_path : str
        Path to MolProbity oneline-analysis binary.
    prepwizard_path : str
        Path to Schrodinger Prepwizard binary.
    max_ligands : int
        Maximum number of heteroatom (i.e. non-protein, non-nucleic, and 
        non-water) residues to permit in a biological assembly (Default: 25).
    retry : bool
        Run as if the code has already been run but did not complete.
    """
    for _dir in [ent_gz_dir, pdb_outdir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    min_molprobity_clusters = []
    for cluster in clusters:
        min_molprobity_clusters.append([])
        for ent in cluster:
            pdb_code = ent[0]
            ent_gz_path = os.path.join(ent_gz_dir, pdb_code[1:3], 
                                       pdb_code + '.ent.gz')
            identifier_chains_dict = parse_compnd(ent_gz_path)
            chid = identifier_chains_dict[ent[1]]
            bio_paths = write_biounits(ent_gz_path, pdb_outdir, max_ligands, 
                                       xtal_only=True, write=(not retry))
            prep_biounits(bio_paths, prepwizard_path)
            scores = [run_molprobity(bio_path, molprobity_path) 
                      for bio_path in bio_paths]
            if min(scores) <= 2.0:
                for bio_path, score in zip(bio_paths, scores):
                    if score == min(scores):
                        biounit = bio_path.split('/')[-1][:-4]
                        for chid in identifier_chains_dict[ent[1]]:
                            min_molprobity_clusters[-1].append((biounit, chid))
        if len(min_molprobity_clusters[-1]) == 0:
            min_molprobity_clusters.pop()
    if len(min_molprobity_clusters):
        with open(final_cluster_outpath, 'w') as f:
            for cluster in min_molprobity_clusters[:-1]:
                for biounit, chid in cluster:
                    f.write(biounit + '_' + chid + ' ')
                f.write('\n')
            for biounit, chid in min_molprobity_clusters[-1]:
                f.write(biounit + '_' + chid + ' ')


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('-j', '--job-index', type=int, default=0, 
                      help="Index for the current job, relevant for multi-job "
                      "HPC runs (Default: 0).")
    argp.add_argument('-n', '--num-jobs', type=int, default=1, 
                      help="Number of jobs, relevant for multi-job HPC runs "
                      "(Default: 1).")
    argp.add_argument('-e', '--ent-gz-dir', help="Path to directory "
                      "containing ent.gz files from which to generate input "
                      "files for COMBS database generation.")
    argp.add_argument('-v', '--validation-dir', help="Path to directory "
                      "containing validation files for PDB structures "
                      "from which to extract resolution and R-free values.")
    argp.add_argument('-c', '--cluster-file', help="Path to file containing "
                      "the RCSB-curated sequence clusters in the PDB.")
    argp.add_argument('-f', '--final-cluster-outpath', default='', 
                      help="Path to output file at which to specify the "
                      "biounits and chains that are the lowest-Molprobity "
                      "score representatives of the RCSB sequence clusters.")
    argp.add_argument('-o', '--pdb-outdir', help="Directory at which to "
                      "output fully prepared PDB files for biological "
                      "assemblies with minimal MolProbity scores.")
    argp.add_argument('--molprobity-path', default='', 
                      help="Path to MolProbity oneline-analysis binary.")
    argp.add_argument('--prepwizard-path', default='',
                      help="Path to Schrodinger Prepwizard binary.")
    argp.add_argument('-m', '--max-ligands', type=int, default=25, 
                      help="Maximum number of heteroatom (i.e. non-protein, "
                      "non-nucleic, and non-water) residues to permit in a "
                      "biological assembly.")
    argp.add_argument('-r', '--retry', action='store_true', 
                      help="Run as if the code has already been run but "
                      "did not complete (i.e. finish generating the files "
                      "that did not generate in an earlier run.")
    return argp.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # determine the sequence clusters of PDB chains to prepare, within which 
    # the elements with the best MolProbity scores will be selected
    clusters = filter_clusters(
        read_clusters(args.cluster_file, args.job_index, args.num_jobs), 
        args.validation_dir
    )
    ent_gz_dir_to_vdg_db_files(args.ent_gz_dir, args.pdb_outdir, 
                               args.final_cluster_outpath, 
                               clusters, args.max_ligands, 
                               args.molprobity_path, args.prepwizard_path, 
                               args.retry)
