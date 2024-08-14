import gzip
import numpy as np
import xml.etree.ElementTree as ET

def extract_global_validation_values(validation_file):
    """Extract resolution and R values from a validation XML file.
    
    Parameters
    ----------
    validation_file : str
        Path to the gzipped validation XML file.

    Returns
    -------
    resolution : float
        Resolution of the structure (in Angstroms).
    r_obs : float
        Observed R value of the structure.
    """
    # Parse the XML file
    with gzip.open(validation_file, 'rt') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Find the resolution and Rfree elements
    resolution = None
    r_obs = None
    try:
        for entry in root.findall(".//Entry"):
            resolution = float(entry.get("PDB-resolution"))
            r_obs = float(entry.get("PDB-R"))
    except:
        pass # no resolution or R-value

    return resolution, r_obs

def extract_residue_validation_values(validation_file, chids_resnums):
    """Extract RSCC, RSR, and RSRZ values from a validation XML file.

    Parameters
    ----------
    validation_file : str
        Path to the gzipped validation XML file.
    chids_resnums : list of tuples
        List of tuples of chain IDs and residue numbers for which to extract 
        the RSCC, RSR, and RSRZ values.

    Returns
    -------
    rscc_values : np.ndarray
        List of RSCC values for the specified residues.
    rsr_values : np.ndarray
        List of RSR values for the specified residues.
    rsrz_values : np.ndarray
        List of RSRZ values for the specified residues.
    """
    # Parse the XML file
    with gzip.open(validation_file, 'rt') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Initialize arrays to store the values
    rscc_values = np.zeros(len(chids_resnums))
    rsr_values = np.zeros(len(chids_resnums))
    rsrz_values = np.zeros(len(chids_resnums))

    # Find the residue elements
    chids, resnums = zip(*chids_resnums)
    for residue in root.findall(".//ModelledSubgroup"):
        chain = residue.get("chain")
        number = int(residue.get("resnum"))
        tup = (chain, number)
        if tup in chids_resnums:
            # Extract RSCC, RSR, and RSRZ values
            try:
                # if tup not in rscc_dict.keys():
                idx = chids_resnums.index(tup)
                rscc_values[idx] = float(residue.get("rscc"))
                rsr_values[idx] = float(residue.get("rsr"))
                rsrz_values[idx] = float(residue.get("rsrz"))
            except:
                print(('Could not extract validation values ' 
                       'for residue {} in chain {}.').format(number, chain))

    return rscc_values, rsr_values, rsrz_values