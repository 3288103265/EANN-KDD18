import random
from transformers import *
import json

filename1 = "/home/wangpenghui/Datasets/tweets/train_rumor.txt"
filename2 = "/home/wangpenghui/Datasets/tweets/train_nonrumor.txt"
filename3 = "/home/wangpenghui/Datasets/tweets/test_rumor.txt"
filename4 = "/home/wangpenghui/Datasets/tweets/test_nonrumor.txt"

def dump_inputs(filename, label):
    """dump file into dataset, [{},{},{}......{}]"""
    dataset = dict()
    file = open(filename, 'r')
    n = 0
    for line in file:
        if n%3==0:
            n+=1
            continue
        elif n%3==1:
            url = line.split('|')[0]
            img_name = url.split('/')[-1]
            n += 1
        else:
            line.rstrip()
            dataset.append([line, label, img_name])
            n += 1

    file.close()

    return dataset


trainset = []
testset = []

trainset = dump_inputs(filename1, 'rumor')
trainset.extend(dump_inputs(filename2, 'nonrumor'))
testset = dump_inputs(filename3, 'rumor')
testset.extend(dump_inputs(filename4, 'nonrumor'))
random.shuffle(trainset)
random.shuffle(testset)

small_trainset = trainset[:256]
small_testset = testset[:128]
json.dump(testset, fp=open('testset.json', 'w'))
json.dump(trainset, fp=open('trainset.json', 'w'))
json.dump(small_trainset, fp=open('small_trainset.json', 'w'))
json.dump(small_testset, fp=open('small_testset.json', 'w'))

# def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, ngram_dict):
#     """Loads a data file into a list of `InputBatch`s."""
#
#     label_map = {label: i for i, label in enumerate(label_list)}
#
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens_a = tokenizer.tokenize(example.text_a)
#
#         tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#         else:
#             # Account for [CLS] and [SEP] with "- 2"
#             if len(tokens_a) > max_seq_length - 2:
#                 tokens_a = tokens_a[:(max_seq_length - 2)]
#
#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids: 0   0   0   0  0     0 0
#         #
#         # Where "type_ids" are used to indicate whether this is the first
#         # sequence or the second sequence. The embedding vectors for `type=0` and
#         # `type=1` were learned during pre-training and are added to the wordpiece
#         # embedding vector (and position vector). This is not *strictly* necessary
#         # since the [SEP] token unambiguously separates the sequences, but it makes
#         # it easier for the model to learn the concept of sequences.
#         #
#         # For classification tasks, the first vector (corresponding to [CLS]) is
#         # used as as the "sentence vector". Note that this only makes sense because
#         # the entire model is fine-tuned.
#         tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#         segment_ids = [0] * len(tokens)
#
#         if tokens_b:
#             tokens += tokens_b + ["[SEP]"]
#             segment_ids += [1] * (len(tokens_b) + 1)
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)
#
#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         label_id = label_map[example.label]
#
#         # ----------- code for ngram BEGIN-----------
#         ngram_matches = []
#         #  Filter the word segment from 2 to 7 to check whether there is a word
#         for p in range(2, 8):
#             for q in range(0, len(tokens) - p + 1):
#                 character_segment = tokens[q:q + p]
#                 # j is the starting position of the word
#                 # i is the length of the current word
#                 character_segment = tuple(character_segment)
#                 if character_segment in ngram_dict.ngram_to_id_dict:
#                     ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
#                     ngram_matches.append([ngram_index, q, p, character_segment])
#
#         shuffle(ngram_matches)
#         # max_word_in_seq_proportion = max_word_in_seq
#         max_word_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
#         if len(ngram_matches) > max_word_in_seq_proportion:
#             ngram_matches = ngram_matches[:max_word_in_seq_proportion]
#         ngram_ids = [ngram[0] for ngram in ngram_matches]
#         ngram_positions = [ngram[1] for ngram in ngram_matches]
#         ngram_lengths = [ngram[2] for ngram in ngram_matches]
#         ngram_tuples = [ngram[3] for ngram in ngram_matches]
#         ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]
#
#         import numpy as np
#         ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
#         ngram_mask_array[:len(ngram_ids)] = 1
#
#         # record the masked positions
#         ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
#         for i in range(len(ngram_ids)):
#             ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0
#
#         # Zero-pad up to the max word in seq length.
#         padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
#         ngram_ids += padding
#         ngram_lengths += padding
#         ngram_seg_ids += padding
#
#         # ----------- code for ngram END-----------
#
#         features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=input_mask,
#                           segment_ids=segment_ids,
#                           label_id=label_id,
#                           ngram_ids=ngram_ids,
#                           ngram_positions=ngram_positions_matrix,
#                           ngram_lengths=ngram_lengths,
#                           ngram_tuples=ngram_tuples,
#                           ngram_seg_ids=ngram_seg_ids,
#                           ngram_masks=ngram_mask_array))
#     return features
