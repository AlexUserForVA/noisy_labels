
from __future__ import print_function

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import dcase_task2.pytorch.net as Net


from dcase_task2.utils.data_tut18_task2 import load_data as load_data_tut18_task2
from dcase_task2.utils.data_tut18_task2 import ID_CLASS_MAPPING as id_class_mapping_tut18_task2

from torchsummary import summary

# seed seed for reproducibility
np.random.seed(4711)


def load_data(data_set, fold, args):
    """ select data """

    if "tut18T2ver" in data_set:
        normalize = "norm" in data_set
        spec_dir = data_set.split("-")[1]
        data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                     train_verified=True, train_unverified=False, normalize=normalize,
                                     fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                     train_file=args.train_file, train_on_all=args.train_on_all,
                                     validate_verified=not args.validate_unverified)
        id_class_mapping = id_class_mapping_tut18_task2

    elif "tut18T2unver" in data_set:
        normalize = "norm" in data_set
        spec_dir = data_set.split("-")[1]
        data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                     train_verified=False, train_unverified=True, normalize=normalize,
                                     fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                     train_file=args.train_file, train_on_all=args.train_on_all,
                                     validate_verified=not args.validate_unverified)
        id_class_mapping = id_class_mapping_tut18_task2

    elif "tut18T2" in data_set:
        normalize = "norm" in data_set
        spec_dir = data_set.split("-")[1]
        data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                     train_verified=True, train_unverified=True, normalize=normalize,
                                     fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                     train_file=args.train_file, train_on_all=args.train_on_all,
                                     validate_verified=not args.validate_unverified)
        id_class_mapping = id_class_mapping_tut18_task2

    return data, id_class_mapping


def get_dump_file_paths(out_path, fold):
    par = 'params.pkl' if fold is None else 'params_%d.pkl' % fold
    log = 'results.pkl' if fold is None else 'results_%d.pkl' % fold
    dump_file = os.path.join(out_path, par)
    log_file = os.path.join(out_path, log)
    return dump_file, log_file


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

    # select model
    # model = select_model(args.model)

    # load data
    # args.data = tut18T2-specs_train_v1
    # args.fold = 1/2/3/4
    print("\nLoading data ...")
    data, _ = load_data(args.data, args.fold, args)
    train_data = data['train']


    # set model dump file
    # print("\nPreparing model ...")
    # out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    # dump_file, log_file = get_dump_file_paths(out_path, args.fold)

    # change parameter dump files

    #if not args.fine_tune:
   #     dump_file = dump_file.replace(".pkl", "_it0.pkl")
   #     log_file = log_file.replace(".pkl", "_it0.pkl")
   #     print("parameter file", dump_file)
   #     print("log file", log_file)

    # labels_plain = torch.from_numpy(dummy)
    # initialize neural network
    vgg_net = Net.Net()
    # train_data, valid_data, test_data = train_set[0], valid_set[0], test_set[0]
    # train_labels, valid_labels, test_labels = train_set[1], valid_set[1], test_set[1]


    # input_plain = torch.from_numpy(train_data)
    # input = input_plain.view(50000, 1, 28, 28)

    # labels_plain = torch.from_numpy(train_labels)

    BATCH_SIZE = 64
    n_samples = train_data.files.size
    NO_OF_BATCHES = n_samples / BATCH_SIZE

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg_net.parameters(), lr=0.01)

    loss_list = []
    acc_list = []

    for epoch in range(200):

        for batch_no in range(NO_OF_BATCHES):
            samples = train_data[batch_no * NO_OF_BATCHES : (batch_no + 1) * NO_OF_BATCHES]
            train_features = samples[0]
            train_label = samples[1]

            torch_train_features = torch.from_numpy(train_features)
            torch_train_label = torch.from_numpy(train_label)

            out = vgg_net(torch_train_features)
            loss = criterion(out, torch_train_label.long())
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch {}: Loss: {:.4f}'.format(epoch + 1, loss.item()))
