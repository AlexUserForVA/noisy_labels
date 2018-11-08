import argparse

from utils.data_tut18_task2 import load_data as load_data_tut18_task2
from utils.data_tut18_task2 import ID_CLASS_MAPPING as id_class_mapping_tut18_task2

def load_data(data_set, fold, args):
    normalize = "norm" in data_set
    spec_dir = data_set.split("-")[1]
    data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                 train_verified=True, train_unverified=True, normalize=normalize,
                                 fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                 train_file=args.train_file, train_on_all=args.train_on_all,
                                 validate_verified=not args.validate_unverified)
    id_class_mapping = id_class_mapping_tut18_task2
    return data, id_class_mapping

if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train audio tagging network.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select model to train.')
    parser.add_argument('--fold', help='train split.', type=int, default=None)
    parser.add_argument('--ini_params', help='path to pretrained parameters.', type=str, default=None)
    parser.add_argument('--tag', help='add tag to result files.', type=str, default=None)
    parser.add_argument('--fine_tune', help='use fine-tune train configuration.', action='store_true')

    # tut18 task2
    parser.add_argument('--train_file', help='train data file.', type=str, default="train.csv")
    parser.add_argument('--max_len', help='maximum spectrogram length.', type=int, default=None)
    parser.add_argument('--min_len', help='minimum spectrogram length.', type=int, default=None)
    parser.add_argument('--no_len_fix', help='do not fix lengths of spectrograms.', action='store_false')
    parser.add_argument('--train_on_all', help='use all files for training.', action='store_true')
    parser.add_argument('--validate_unverified', help='validate also on unverified samples.', action='store_true')

    args = parser.parse_args()

    data, _ = load_data(args.data, args.fold, args)

    for e in range(5):
        X, y = data['train'][e]
        print(X.shape)
        print(y.shape)