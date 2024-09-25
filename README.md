# vdG-miner

## Credits

This extension builds upon https://github.com/degrado-lab/vdG-miner by Rian Kormos. 

## New Features

- __Support for Custom PDB Databases__: Adds functionality for using custom PDB databases. While vdG-miner was originally designed to extract van der Graphs of amino acid-derived functional groups (as defined by Polizzi & DeGrado 2020), different approaches are necessary for small molecule-derived functional groups. For more information on these differences, see SMARTS-vdg (TODO: insert link)
- __Simplified Dependencies__: Removes requirement of supplying PDB validation reports and water-containing PDBs.

## Usage

The scripts in this fork have been modified to be used within [ligand-vdGs](https://github.com/tan-sophia/ligand-vdGs.git). Please refer to that top-level package for usage instructions.