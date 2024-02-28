# vdG-miner
The full pipeline for extracting vdGs from a mirror of the PDB.

### using the code
This code is intended to generate the necessary pickled objects for the generation of van der Graph 
(vdG) databases to aid in the design of ligand-binding proteins.  A mirror of the PDB and validation 
information for each structure is a necessary prerequisite.  These can be downloaded from the 
PDB FTP server as follows:

```bash
> rsync -rlpt -v -z --delete --port=33444
  rsync.rcsb.org::ftp_data/structures/divided/pdb/ $LOCAL_PDB_MIRROR_PATH
  
> rsync -rlpt -v -z --delete --include="*/" --include="*.xml.gz" --exclude="*"
  --port=33444 rsync.rcsb.org::ftp/validation_reports/ $LOCAL_VALIDATION_PATH
```

### calculating 3Di sequences
The 3Di module of this library can rapidy calculate 3Di sequences (van Kempen et al., 2023) for 
input PDB files with one or more protein chains. Example usage is:

```bash
> vdg_miner/3Di.py encoder.pkl PDB_file.pdb
```