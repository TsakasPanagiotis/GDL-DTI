import pickle
from dataclasses import dataclass


@dataclass
class RawDataPaths:
    b_values_file: str = 'raw_data/mgh_1001/data/bvals.txt'
    b_vectors_file: str = 'raw_data/mgh_1001/data/bvecs_moco_norm.txt'
    raw_data_file: str = 'raw_data/mgh_1001/data/diff_preproc.nii.gz'


def main():
    paths = RawDataPaths()

    with open('raw_data/mgh_1001/paths.pkl', 'wb') as f:
        pickle.dump(paths, f)


if __name__ == '__main__':
    main()
