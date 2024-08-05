import os
import sys
import argparse

import numpy as np
import prody as pr
from tqdm import tqdm

def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('cg_atoms', type=str, nargs='+', 
                      help="Names of CG atoms from which to form a reference "
                           "frame such that, for atoms A, B, and C, the "
                           "bisector of the A-B-C angle is the x-axis, the "
                           "vector perpendicular to the ABC plane is the "
                           "z-axis, and the y-axis is right-handed.")
    argp.add_argument('-t', '--threshold', type=int, default=5, 
                      help="The threshold cluster size to extract.")
    argp.add_argument('-o', '--outdir', type=os.path.realpath, default='.', 
                      help="Path to directory at which to output gzipped "
                           "PDB files containing the aligned vdMs in their "
                           "native structural contexts. If no such directoy "
                           "exists, it will be created.")
    return argp.parse_args()

def kabsch(X, Y):
    n = len(X)
    Xbar, Ybar = np.mean(X, axis=0), np.mean(Y, axis=0)
    Xc, Yc = X - Xbar, Y - Ybar
    H = np.dot(Xc.T, Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.dot(U, Vt)))
    D = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., d]])
    R = np.dot(U, np.dot(D, Vt))
    t = Ybar - np.dot(Xbar, R)
    # print(np.sqrt(((X @ R + t - Y) ** 2).sum() / n))
    return R, t

def coords_to_frame(coords):
    frame = np.zeros((3, 3))
    frame[0] = coords[0] - 2. * coords[1] + coords[2]
    frame[0] /= np.linalg.norm(frame[0])
    frame[2] = np.cross(coords[0] - coords[1], coords[2] - coords[1])
    frame[2] /= np.linalg.norm(frame[2])
    frame[1] = np.cross(frame[2], frame[0])
    return frame

def extract_frame_prody(struct, scr, atom_names):
    coords = np.zeros((3, 3))
    base_selstr = 'segment {} and chain {} and resnum {}'.format(*scr)
    for i, name in enumerate(atom_names):
        selstr = base_selstr + ' and name ' + name
        coords[i] = struct.select(selstr).getCoords()
    return coords_to_frame(coords), coords[1]

def load_fingerprints_and_envs(threshold=5):
    parent_dir = "/wynton/group/degradolab/nr_pdb/"
    clusters = np.load(os.path.join(parent_dir, "clusters_4.npy"))
    clusters_unique, inverse, counts = np.unique(clusters, 
                                                 return_inverse=True, 
                                                 return_counts=True)
    sort_idxs = np.argsort(counts)[::-1]
    sort_clusters = np.argsort(clusters_unique[sort_idxs])
    all_counts = counts[inverse]
    all_clusters = clusters_unique[inverse]
    n_gt_thresh = (counts >= threshold).sum()
    fingerprints = [np.empty((0, 2899), dtype=np.bool_) 
                    for i in range(n_gt_thresh)]
    environments = [[] for i in range(n_gt_thresh)]
    counter = 0
    for subdir in tqdm(os.listdir(os.path.join(parent_dir, 'fingerprints'))):
        for file in os.listdir(
                    os.path.join(
                        parent_dir, 
                        'fingerprints', 
                        subdir
                    )
                ):
            if file.endswith('_fingerprints.npy'):
                fingerprint_array = np.load(
                    os.path.join(
                        parent_dir, 
                        'fingerprints', 
                        subdir, 
                        file
                    )
                )
                with open(
                        os.path.join(
                            parent_dir, 
                            'fingerprints', 
                            subdir, 
                            file.replace(
                                '_fingerprints.npy', 
                                '_environments.txt')
                            ), 
                            'r'
                        ) as f:
                    for line, fingerprint in zip(f.readlines(), 
                                                 fingerprint_array):
                        if all_counts[counter] >= threshold:
                            reordered_cluster = \
                                sort_clusters[all_clusters[counter]]
                            fingerprints[reordered_cluster] = \
                                np.vstack(
                                    (fingerprints[reordered_cluster], 
                                    fingerprint)
                            )
                            environments[reordered_cluster].append(
                                eval(line.strip())
                            )
                        counter += 1
    return fingerprints, environments

def main(cg_atoms, threshold, outdir):
    pdb_dir = "/wynton/group/degradolab/nr_pdb/clean_final_pdb/"
    os.makedirs(outdir, exist_ok=True)
    fingerprints, envs = load_fingerprints_and_envs(threshold)
    for k, env_cluster in enumerate(envs):
        len_cluster = len(env_cluster)
        n_residues = len(env_cluster[0]) - 1
        cluster_name = 'cluster_{}_numenv_{}_size_{}'.format(k, len_cluster, 
                                                             n_residues)
        os.makedirs(os.path.join(outdir, cluster_name), exist_ok=True)
        for i, env in enumerate(env_cluster):
            flag = False
            biounit = env[0][0]
            middle_two = biounit[1:3].lower()
            pdb_file = os.path.join(pdb_dir, middle_two, biounit + '.pdb')
            struct = pr.parsePDB(pdb_file)
            scrs = [(tup[1], tup[2], tup[3]) for tup in env]
            for scr in scrs:
                selstr = 'segment {} and chain {} and resnum {}'.format(*scr)
                try:
                    struct.select(selstr).setOccupancies(2.0)
                except:
                    print('Bad SCR: ', scr)
                    flag = True
            try:
                selstr0 = 'segment {} and chain {} and resnum {}'.format(
                    *scrs[0]
                )
                frame, pos = extract_frame_prody(struct, scrs[0], cg_atoms)
            except:
                print('Bad Env/SCR: ', env[0])
                flag = True
            if flag:
                continue
            struct.setCoords((struct.getCoords() - pos) @ frame.T)
            selstr_radius = ('same residue as '
                             '(within 15 of ({}))').format(selstr0)
            pr.writePDB(os.path.join(outdir, cluster_name, biounit + '.pdb'),
                        struct.select(selstr_radius), secondary=False)


if __name__ == "__main__":
    args = parse_args()
    main(args.cg_atoms, args.threshold, args.outdir)