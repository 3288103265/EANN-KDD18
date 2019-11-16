import math
import os
import pickle
import random
import json
import numpy as np
import torch

from ZEN import BertTokenizer, ZenNgramDict
from torch.utils.data import Subset
from torchvision.transforms import transforms

from utils_multi_dataset import FusionFolder

filename1 = "/home/wangpenghui/Datasets/tweets/train_rumor.txt"
filename2 = "/home/wangpenghui/Datasets/tweets/train_nonrumor.txt"
filename3 = "/home/wangpenghui/Datasets/tweets/test_rumor.txt"
filename4 = "/home/wangpenghui/Datasets/tweets/test_nonrumor.txt"
non_rumor_path = "/home/wangpenghui/Datasets/WeiboRumorSet/nonrumor_images/"
rumor_path = "/home/wangpenghui/Datasets/WeiboRumorSet/rumor_images/"

# TODO: update tk and ngram_dict
tokenizer = BertTokenizer.from_pretrained('/home/wangpenghui/EANN-KDD18/ZEN-master/models/output/checkpoint-4500',
                                          do_lower_case=True)
ngram_dict = ZenNgramDict('/home/wangpenghui/EANN-KDD18/ZEN-master/models/output/checkpoint-4500', tokenizer=tokenizer)


def convert_input_to_items(line, max_seq_length=384):
    """input sentences,
    outpyt: [input_id, ngram_id, ngram_position]"""
    tokens_a = tokenizer.tokenize(line)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding

    assert len(input_ids) == max_seq_length

    # ----------- code for ngram BEGIN-----------
    ngram_matches = []
    #  Filter the word segment from 2 to 7 to check whether there is a word
    for p in range(2, 8):
        for q in range(0, len(tokens) - p + 1):
            character_segment = tokens[q:q + p]
            # j is the starting position of the word
            # i is the length of the current word
            character_segment = tuple(character_segment)
            if character_segment in ngram_dict.ngram_to_id_dict:
                ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                ngram_matches.append([ngram_index, q, p, character_segment])

    random.shuffle(ngram_matches)
    # max_word_in_seq_proportion = max_word_in_seq
    max_word_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
    if len(ngram_matches) > max_word_in_seq_proportion:
        ngram_matches = ngram_matches[:max_word_in_seq_proportion]
    ngram_ids = [ngram[0] for ngram in ngram_matches]
    ngram_positions = [ngram[1] for ngram in ngram_matches]
    ngram_lengths = [ngram[2] for ngram in ngram_matches]

    ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
    ngram_mask_array[:len(ngram_ids)] = 1

    # record the masked positions
    ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
    for i in range(len(ngram_ids)):
        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

    padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
    ngram_ids += padding

    out = []
    out.append(np.array(input_ids))
    out.append(np.array(ngram_ids))
    out.append(ngram_positions_matrix)

    return out


def dump_inputs(filename, label):
    """dump file into dataset"""
    dataset = dict()
    file = open(filename, 'r')
    num_line = 0
    for line in file:
        if num_line % 3 == 0:
            num_line += 1
            continue
        elif num_line % 3 == 1:
            # add all img_path in every line into a list.
            url_list = line.split('|')
            img_path_list = []
            for url in url_list:
                if url == 'null\n':
                    break
                img_name = url.split('/')[-1]
                if label == 'nonrumor':
                    img_path = os.path.join(non_rumor_path, img_name)
                elif label == 'rumor':
                    img_path = os.path.join(rumor_path, img_name)
                else:
                    raise RuntimeError('unknown labels.')

                img_path_list.append(img_path)
            num_line += 1
        else:
            line.rstrip()
            item = convert_input_to_items(line)
            for path in img_path_list:
                dataset[path] = item
            num_line += 1

    file.close()

    return dataset


# # return all_dict.pkl.. Fixed!!!
def get_all_dict():
    trainset_dict = dump_inputs(filename1, 'rumor')
    trainset_dict.update(dump_inputs(filename2, 'nonrumor'))
    testset_dict = dump_inputs(filename3, 'rumor')
    testset_dict.update(dump_inputs(filename4, 'nonrumor'))
    trainset_dict.update(testset_dict)
    F = open('all_dict.pkl', 'wb')
    pickle.dump(trainset_dict, F)
    F.close()


def get_img_path_list(filename, label):
    file = open(filename, 'r')
    img_path_list = []
    num_line = 0
    for line in file:
        if num_line % 3 == 1:
            # add all img_path in every line into a list.
            url_list = line.split('|')
            for url in url_list:
                if url == 'null\n':
                    break
                img_name = url.split('/')[-1]
                if label == 'nonrumor':
                    img_path = os.path.join(non_rumor_path, img_name)
                elif label == 'rumor':
                    img_path = os.path.join(rumor_path, img_name)
                else:
                    raise RuntimeError('unknown labels.')

                img_path_list.append(img_path)
            num_line += 1
        else:
            num_line += 1
            continue

    file.close()

    return img_path_list


# get train_set.pkl and test_set.pkl
def get_dataset():
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    F = open('all_dict.pkl', 'rb')
    all_dict = pickle.load(F)
    F.close()
    all_sets = FusionFolder("/home/wangpenghui/Datasets/WeiboRumorSet/", all_dict, transform=data_transforms)

    # split dataset into trainset and test set.
    # # Do not random_split dataset.!!
    # train_size = int(0.6 * len(all_sets))
    # test_size = len(all_sets) - train_size
    # torch.manual_seed(1000)
    # train_dataset, test_dataset = torch.utils.data.random_split(all_sets, [train_size, test_size])

    train_img_path_list = get_img_path_list(filename1, 'rumor')
    train_img_path_list.extend(get_img_path_list(filename2, 'nonrumor'))
    test_img_path_list = get_img_path_list(filename3, 'rumor')
    test_img_path_list.extend(get_img_path_list(filename4, 'nonrumor'))

    train_split_index = []
    test_split_index = []
    for idx, (path, _) in enumerate(all_sets.imgs):
        if path in train_img_path_list:
            train_split_index.append(idx)
        elif path in test_img_path_list:
            test_split_index.append(idx)
        else:
            raise RuntimeError('Path not in the list')

    train_dataset = Subset(all_sets, train_split_index)
    test_dataset = Subset(all_sets, test_split_index)

    F_train = open('train_set.pkl', 'wb')
    pickle.dump(train_dataset, F_train)
    F_train.close()

    F_test = open('test_set.pkl', 'wb')
    pickle.dump(test_dataset, F_test)
    F_test.close()


get_dataset()
