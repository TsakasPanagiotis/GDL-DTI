'''[MGH HCP Adult Diffusion](https://db.humanconnectome.org/data/projects/MGH_DIFF) | MGH_1001'''


import pickle


class RawDataPaths:
    def __init__(self) -> None:
        self.b_values_file = 'raw_data/mgh_1001/data/bvals.txt'
        self.b_vectors_file = 'raw_data/mgh_1001/data/bvecs_moco_norm.txt'
        self.raw_data_file = 'raw_data/mgh_1001/data/diff_preproc.nii.gz'
        self.paths_file = 'raw_data/mgh_1001/paths.pkl'


def main():
    paths = RawDataPaths()

    with open(paths.paths_file, 'wb') as f:
        pickle.dump(paths, f)


if __name__ == '__main__':
    main()
