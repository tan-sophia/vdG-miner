import os
from openbabel import openbabel as ob

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def find_cg_matches(smarts_pattern, pdb_path, include_water=False):
    """
    Find CGs matching a SMARTS pattern in PDB files.
    
    Parameters
    ----------
    smarts_pattern : str
        SMARTS pattern.
    pdb_path : str
        Path to directory containing PDB files organized in subdirectories by
        the middle two characters of the PDB ID.
    include_water : bool, optional
        Whether to include water molecules in the search. Default is False.
    
    Returns
    -------
    matches : dict
        Dictionary of matching CGs in ligands with keys as tuples of 
        (seg, chain, resnum, resname) for the ligand and values as lists 
        of atom indices.
    """
    biounit = pdb_path.split('/')[-1][:-4]
    # Read PDB file and extract ligands as blocks
    ligands = {}
    atom_nums = {}
    with open(pdb_path, 'rb') as f:
        for b_line in f:
            if b_line.startswith(b'HETATM'):
                line = b_line.decode('utf-8')
                resname = line[17:20].strip()
                if not include_water and resname == 'HOH':
                    continue
                seg = line[72:76].strip()
                chain = line[21]
                resnum = line[23:26].strip()
                atom_num = line[6:11].strip()
                key = (biounit, seg, chain, resnum, resname)
                if key not in ligands.keys():
                    ligands[key] = line
                else:
                    ligands[key] += line
                atom_nums[atom_num] = key
            if b_line.startswith(b'CONECT'):
                line = b_line.decode('utf-8')
                atom = line.split()[1]
                if atom in atom_nums.keys():
                    ligands[atom_nums[atom]] += line

    # Initialize Open Babel conversion object
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("pdb")

    # Initialize Open Babel SMARTS matcher
    smarts = ob.OBSmartsPattern()
    smarts.Init(smarts_pattern)
    
    # Find CGs matching SMARTS pattern
    matches = {}
    with suppress_stdout_stderr():
        for key, block in ligands.items():
            # Read ligand block as OBMol object
            mol = ob.OBMol()
            obConversion.ReadString(mol, block)
            mol.PerceiveBondOrders()
            # Match SMARTS pattern to ligand
            if smarts.Match(mol):
                atom_names = [line[12:16].strip() 
                            for line in ligands[key].split('\n')
                            if line.startswith('HETATM')]
                matching_atoms = smarts.GetUMapList()
                for match in matching_atoms:
                    try:
                        match_names = [atom_names[i - 1] for i in match]
                        atom_types = [mol.GetAtom(i).GetType() for i in match]
                        match_atoms = list(zip(match_names, atom_types))
                        if key not in matches.keys():
                            matches[key] = [match_atoms]
                        else:
                            matches[key].append(match_atoms)
                    except:
                        pass

    return matches