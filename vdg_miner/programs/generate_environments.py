"""Generate contact-defined vdG environments without legacy feature vectors."""

import argparse
import json
import os
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../vdg"))
from vdg import VDG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate contact-defined vdG environments for PDB structures.")
    parser.add_argument("-c", "--cg", required=True,
                        help="Chemical-group label.")
    parser.add_argument("-m", "--cg-match-dict-pkl", required=True,
                        help="SMARTS match dictionary from smarts_to_cgs.py.")
    parser.add_argument("-p", "--pdb-dir", required=True,
                        help="Reduced PDB mirror.")
    parser.add_argument("-b", "--probe-dir", required=True,
                        help="Probe-output mirror.")
    parser.add_argument("-v", "--validation-dir", default="",
                        help="Validation-report mirror (currently unused).")
    parser.add_argument("-o", "--outdir", required=True,
                        help="Directory for environment JSONL shards.")
    parser.add_argument("-j", "--job-index", type=int, default=0,
                        help="Zero-based worker index.")
    parser.add_argument("-n", "--num-jobs", type=int, default=1,
                        help="Number of structure partitions.")
    return parser.parse_args()


def _json_default(value):
    # ProDy/NumPy residue numbers can be NumPy scalars.
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def main():
    args = parse_args()
    with open(args.cg_match_dict_pkl, "rb") as handle:
        cg_match_dict = pickle.load(handle)
    if not cg_match_dict:
        return

    first_matches = next(iter(cg_match_dict.values()))
    cg_natoms = len(first_matches[0])
    vdg = VDG(args.cg, pdb_dir=args.pdb_dir, probe_dir=args.probe_dir,
              validation_dir=args.validation_dir, cg_natoms=cg_natoms)
    environments_dir = os.path.join(args.outdir, "environments")
    os.makedirs(environments_dir, exist_ok=True)

    # Grouped in one pass. Re-scanning the whole match dict per structure is
    # quadratic, and for a common CG both factors run into the tens of thousands.
    matches_by_structure = {}
    for key, value in cg_match_dict.items():
        matches_by_structure.setdefault(key[0], {})[key] = value

    structures = sorted(matches_by_structure)
    for structure_index, structure in enumerate(structures):
        if structure_index % args.num_jobs != args.job_index:
            continue
        structure_matches = matches_by_structure[structure]
        # Passed explicitly: every CG here is a ligand fragment, not part of the
        # chain, so sequence separation from the CG residue is not defined and
        # the filter must stay off. See VDG.mine_environments' docstring.
        environments = vdg.mine_environments(cg_match_dict=structure_matches,
                                             min_seq_sep=1)
        if not environments:
            continue
        middle_two = structure[1:3].lower()
        shard_dir = os.path.join(environments_dir, middle_two)
        os.makedirs(shard_dir, exist_ok=True)
        shard_path = os.path.join(
            shard_dir, f"{structure}__{args.job_index:04d}.jsonl")
        with open(shard_path, "w") as handle:
            for environment in environments:
                json.dump(environment, handle, default=_json_default,
                          separators=(",", ":"))
                handle.write("\n")


if __name__ == "__main__":
    main()
