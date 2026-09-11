import os
import re
import sys
from openbabel import openbabel as ob
from ligand_vdgs.functions import ligand_perception, parent_db
import time


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
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def read_ligand_blocks(pdb_path, include_water=False):
    """{biounit, seg, chain, resnum, resname} -> PDB text block, per HETATM residue.

    Split out of find_cg_matches so a caller matching many SMARTS against one
    structure parses and perceives it once instead of once per pattern. Returns
    None when the file cannot be read after retries; callers distinguish that
    from a structure with no ligands, which is an empty dict.
    """
    biounit = parent_db.stem_of(pdb_path)
    ligands = {}
    atom_nums = {}

    # Retries are for transient filesystem hiccups on Wynton, not for bad
    # input; a genuinely unreadable file should fail in ~2 min, not hours.
    num_attempts = 5
    retry_delay = 30
    for attempt in range(num_attempts):
        # A failed attempt may have appended part of the file already.
        ligands = {}
        atom_nums = {}
        try:
            with open(pdb_path, 'rb') as f:
                for b_line in f:
                    if b_line.startswith(b'HETATM'):
                        line = b_line.decode('utf-8')
                        resname = line[17:20].strip()
                        if not include_water and resname == 'HOH':
                            continue
                        seg = line[72:76].strip()
                        chain = line[21]
                        resnum = line[22:26].strip()
                        atom_num = line[6:11].strip()
                        key = (biounit, seg, chain, resnum, resname)
                        if key not in ligands.keys():
                            ligands[key] = line
                        else:
                            ligands[key] += line
                        atom_nums[atom_num] = key
                    if b_line.startswith(b'CONECT'):
                        line = b_line.decode('utf-8')
                        atom0 = line[6:11].strip()
                        if atom0 in atom_nums.keys():
                            line_to_add = 'CONECT' + atom0.rjust(5)
                            for atom in line.split()[2:]:
                                if atom in atom_nums.keys():
                                    line_to_add += atom.rjust(5)
                            if len(line_to_add) > 11:
                                ligands[atom_nums[atom0]] += line
            return ligands
        except Exception as e:
            if attempt < num_attempts - 1:
                time.sleep(retry_delay)
            else:
                print(f'vdG-miner/vdg_miner/vdg/cg.py could not read {pdb_path}: {e}')
                return None


# SMARTS ``r<n>`` does not mean the same thing in the two toolkits ligand-vdGs
# uses. RDKit reads it as "the atom's SMALLEST ring has size n"; OpenBabel reads
# it as "the atom is a member of SOME n-ring". Fragment keys are written by RDKit
# (ligand_vdgs/functions/Frags.py ring_query_for_atom) and matched against query
# ligands by RDKit at hit-finding time, but matched against database ligands by
# OpenBabel here -- so without this filter the miner accepts CG instances that hit
# finding can never reproduce. Measured on the shipped fragment dict: 4292 of 6144
# keys carry r<n>, and the two toolkits return different match sets for 3.2% of
# them, in both directions (spiro/fused/bridged systems with mixed ring sizes).
#
# OBAtom::MemberOfRingSize() returns the smallest ring the atom belongs to, which
# is exactly RDKit's reading, so the fix is to re-impose it on OpenBabel's matches
# rather than to change the key syntax.
_ORGANIC_SUBSET = ('Cl', 'Br', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'I',
                   'b', 'c', 'n', 'o', 'p', 's', '*')


def ring_size_constraints(smarts_pattern):
    """Required smallest-ring size per pattern atom, or None where unconstrained.

    Returned in pattern-atom order, i.e. the order OBSmartsPattern's match tuples
    use. Only the narrow SMARTS grammar ligand-vdGs emits is handled (bracket
    atoms, organic-subset atoms, bonds, branches, ring-closure digits); the caller
    checks the length against OBSmartsPattern.NumAtoms() so an unparsed construct
    fails loudly instead of silently disabling the filter.
    """
    constraints = []
    i, n = 0, len(smarts_pattern)
    while i < n:
        char = smarts_pattern[i]
        if char == '[':
            depth, j = 1, i + 1
            while j < n and depth:
                if smarts_pattern[j] == '[':
                    depth += 1
                elif smarts_pattern[j] == ']':
                    depth -= 1
                j += 1
            primitives = smarts_pattern[i + 1:j - 1]
            # Only a positive, unnegated r<n> constrains anything; "!r5" is a
            # negation and is left to OpenBabel, where it means the same thing.
            match = re.search(r'(?<!!)\br(\d+)', ';' + primitives.replace(',', ';'))
            constraints.append(int(match.group(1)) if match else None)
            i = j
            continue
        for symbol in _ORGANIC_SUBSET:
            if smarts_pattern.startswith(symbol, i):
                constraints.append(None)
                i += len(symbol)
                break
        else:
            # Bond, branch, ring-closure digit, %nn, or dot: not an atom.
            i += 2 if char == '%' else 1
    return constraints


def match_satisfies_ring_sizes(mol, match, constraints):
    """Whether every r<n>-annotated position of `match` has smallest ring size n."""
    for atom_idx, required in zip(match, constraints):
        if required is None:
            continue
        atom = mol.GetAtom(atom_idx)
        if atom is None or atom.MemberOfRingSize() != required:
            return False
    return True


def check_ring_constraints(smarts_pattern, pattern, constraints):
    """Raise if the tokenizer disagrees with OpenBabel on the atom count."""
    if len(constraints) != pattern.NumAtoms():
        raise ValueError(
            f"ring-size constraint parse produced {len(constraints)} atoms for "
            f"{smarts_pattern!r}, but OpenBabel parsed {pattern.NumAtoms()}; "
            "refusing to mine with the r<n> filter disabled")


def compile_smarts_patterns(smarts_list):
    """Compile SMARTS once, for reuse across many structures.

    Compilation is not free at scale: a caller scanning a parent database with a couple
    of thousand patterns that rebuilt them per structure spent nearly all of its
    time in OBSmartsPattern.Init rather than matching. Compile once, pass the
    result to count_matching_structures.
    """
    patterns = []
    for smarts in smarts_list:
        pattern = ob.OBSmartsPattern()
        if not pattern.Init(smarts):
            raise ValueError(f"OpenBabel could not parse SMARTS: {smarts!r}")
        patterns.append(pattern)
    return patterns


def count_matching_structures(smarts_patterns, pdb_path, include_water=False):
    """Indices of `smarts_patterns` that match at least one ligand in this PDB.

    The point is the shared parse: OpenBabel reads and perceives each ligand
    block once, then every pattern is matched against the same OBMol. Used to
    size jobs, where only presence matters, so it stops at the first hit per
    pattern and never builds atom names.

    `smarts_patterns` may be SMARTS strings or the output of
    compile_smarts_patterns; pass the latter when scanning many structures.
    """
    ligands = read_ligand_blocks(pdb_path, include_water=include_water)
    if not ligands:
        return set()
    patterns = (smarts_patterns
                if smarts_patterns and isinstance(smarts_patterns[0], ob.OBSmartsPattern)
                else compile_smarts_patterns(smarts_patterns))
    hits = set()
    with suppress_stdout_stderr():
        for key, block in ligands.items():
            perceived = ligand_perception.perceive_ligand_instance(block, key[-1])
            if perceived is None:
                continue
            mol = perceived.obmol
            # A templated mol carries phantom atoms for unobserved density.
            # A match reaching one is never mined (find_cg_matches drops it),
            # so counting it here would inflate the presence count a fragment
            # is admitted on.
            named = ligand_perception.pdb_atom_names(mol)
            for i, pattern in enumerate(patterns):
                if i in hits or not pattern.Match(mol):
                    continue
                if any(all(idx in named for idx in m)
                       for m in pattern.GetUMapList()):
                    hits.add(i)
            if len(hits) == len(patterns):
                break
    return hits


# Value written into an int8 annotation column when the perception could not
# supply it. Non-negative codes are real values; `perception` uses its own
# non-negative scale, so the two never collide.
ANNOT_UNREADABLE = -1


def _cg_atom_annotations(mol, match):
    """Per-CG-atom chemistry for one SMARTS match, in match (SMARTS-slot) order.

    Everything here is metadata, not key material: heavy degree and the carbon
    H flag are already in the key, while these are the finer values a read path
    may want to pool on later (rebuild-notes 1). `nbr_elems` describes the
    atoms the fragment was cut away from, so it is per observation and never a
    key primitive.

    Never raises. `smarts_to_cgs.py` retries any exception from this call 100
    times with 100-second sleeps, so an odd ligand must degrade to the
    unreadable sentinel rather than stall a whole job.
    """
    in_match = set(match)
    degrees, num_hs, charges, nbr_elems = [], [], [], []
    for idx in match:
        # Every value for one atom is computed before anything is appended, so a
        # failure part-way cannot leave the four lists at different lengths --
        # which would silently shift every later atom's annotation by one.
        try:
            atom = mol.GetAtom(idx)
            neighbors = list(ob.OBAtomAtomIter(atom))
            heavy = [nbr for nbr in neighbors if nbr.GetAtomicNum() != 1]
            # The mol is hydrogen-free by the time it gets here, so the H count
            # is implicit; the explicit term is for a caller that skipped the
            # strip. This is OpenBabel's placed-hydrogen view, i.e.
            # protonation-bound -- which is what the perception column records.
            n_h = (atom.GetImplicitHCount()
                   + sum(1 for nbr in neighbors if nbr.GetAtomicNum() == 1))
            values = (len(heavy), n_h, atom.GetFormalCharge(),
                      ''.join(sorted(ob.GetSymbol(nbr.GetAtomicNum())
                                     for nbr in heavy
                                     if nbr.GetIdx() not in in_match)))
        except Exception:
            values = (ANNOT_UNREADABLE, ANNOT_UNREADABLE, ANNOT_UNREADABLE, '')
        degrees.append(values[0])
        num_hs.append(values[1])
        charges.append(values[2])
        nbr_elems.append(values[3])
    return {'heavy_degree': degrees, 'num_h': num_hs,
            'formal_charge': charges, 'nbr_elems': nbr_elems}


def find_cg_matches(smarts_pattern, pdb_path,
                    include_water=False, return_mol_objs=False,
                    return_annotations=False):
    """
    Find CGs matching a SMARTS pattern in PDB files.
    
    Parameters
    ----------
    smarts_pattern : str
        SMARTS pattern.
    pdb_path : str
        Path to directory containing PDB files organized in subdirectories 
        by the middle two characters of the PDB ID.
    include_water : bool, optional
        Whether to include water molecules in the search. Default is False.
    
    Returns
    -------
    return_annotations : bool, optional
        Also return `cg_annot_dict`, a structure parallel to `cg_match_dict`:
        same keys, same list positions, each entry the per-CG-atom annotations
        and the perception code for that match. Kept separate rather than
        folded into `cg_match_dict` because `generate_environments.py` and
        `fingerprints_to_pdbs.py` both index the match lists as plain name
        lists; changing their shape would break those readers.

    Returns
    -------
    cg_match_dict : dict, optional
        Dictionary of matching CGs in ligands, with tuples of 
        (struct_name, seg, chain, resnum, resname) for the ligand as keys 
        and tuples that pair the list of atom names and the list of contacting 
        chains for each match to the CG as values. Used for non-protein CGs.
    """
    ligands = read_ligand_blocks(pdb_path, include_water=include_water)
    if ligands is None:
        return {}, {}


    # Initialize Open Babel SMARTS matcher
    smarts = ob.OBSmartsPattern()
    if not smarts.Init(smarts_pattern):
        raise ValueError(f"OpenBabel could not parse SMARTS: {smarts_pattern!r}")
    # See ring_size_constraints: OpenBabel's r<n> is looser than the RDKit r<n>
    # these keys were written with, so its matches are re-filtered below.
    ring_constraints = ring_size_constraints(smarts_pattern)
    check_ring_constraints(smarts_pattern, smarts, ring_constraints)
    
    # Store mol objects that contain the SMARTS pattern
    match_mol_objs = {}
    
    # Find CGs matching SMARTS pattern
    cg_match_dict = {}
    cg_annot_dict = {}
    with suppress_stdout_stderr():
        for key, block in ligands.items():
            ligname = key[-1]
            # Chemistry and the H-free graph both come from the one perception
            # entry point, so the CCD swap reaches the matcher and the key
            # writer together. Every index below -- the name map, the
            # GetUMapList tuples, the returned match_mol_objs -- is an index
            # into this mol.
            perceived = ligand_perception.perceive_ligand_instance(block, ligname)
            if perceived is None:
                continue
            mol = perceived.obmol
            if smarts.Match(mol):
                matches = [m for m in smarts.GetUMapList()
                           if match_satisfies_ring_sizes(mol, m, ring_constraints)]
                # Filtered before match_mol_objs, so a ligand whose only matches
                # fail the ring-size check is not reported as a matching mol.
                if not matches:
                    continue
                if ligname not in match_mol_objs.keys():
                    match_mol_objs[ligname] = mol
                atom_names = ligand_perception.pdb_atom_names(mol)
                for match in matches:
                    # Atoms obabel perceived rather than read have no PDB name.
                    if any(i not in atom_names for i in match):
                        continue
                    cg_match_dict.setdefault(key, []).append(
                        [atom_names[i] for i in match])
                    if return_annotations:
                        # Appended in the same branch and the same order as the
                        # names above, so position i of one list describes
                        # position i of the other. Any `continue` that skips one
                        # must skip both.
                        annot = _cg_atom_annotations(mol, match)
                        annot['perception'] = perceived.provenance
                        cg_annot_dict.setdefault(key, []).append(annot)

    result = (cg_match_dict, match_mol_objs if return_mol_objs else {})
    return result + (cg_annot_dict,) if return_annotations else result
