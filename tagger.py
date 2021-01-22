"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchtext import data
import torch.optim as optim
from math import log, isfinite
from collections import Counter

import sys, os, time, platform, nltk, random
import numpy as np

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed = 1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    #torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'Rotem Lev Lehman', 'id1': '208965814', 'email1': 'levlerot@post.bgu.ac.il',
            'name2': 'Koren Gershoni', 'id2': '311272264', 'email2': 'korenger@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities


def insert2dict_of_counters(d, key1, key2):
    """ Adds 1 to d[key1][key2]. If d[key1] does not exist, creates a new counter for it and then add 1.

    :param d: the dictionary
    :type d: dict
    :param key1: the outer key
    :type key1: str
    :param key2: the inner key
    :type key2: str
    """
    if key1 not in d.keys():
        d[key1] = Counter()
    d[key1][key2] += 1


def get_probabilities_dict(d):
    """ Creates a log-probability dictionary from a counter dictionary.

    :param d: the counter dictionary
    :type d: dict
    :return: a log-probability dictionary corresponding to d
    :rtype: dict
    """
    prob_dict = {}
    for key1 in d.keys():
        ctr = d[key1]
        total = sum(ctr.values())
        prob_dict[key1] = {key2: log(count / total) for key2, count in ctr.items()}
    return prob_dict


def smooth_dictionary(d, smooth_options):
    """ Smoothed the given counter dictionary. For each entry, if some smooth-option does not exist in it, put 1 in it,
    And for the entries that do exist in it, add 1 to them. This will create a Laplace-smoothing as we wanted.

    :param d: the dictionary to smooth
    :type d: dict
    :param smooth_options: the smooth options. In each entry of d, there must be all of these options.
    :type smooth_options: set
    """
    for key1 in d.keys():
        ctr = d[key1]
        if key1 == START:
            # special case - we don't allow a transition from the START immediately to the END
            curr_smooth_options = smooth_options - {END}
        else:
            curr_smooth_options = smooth_options
        for key2 in curr_smooth_options:
            ctr[key2] += 1


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as returned by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    global perWordTagCounts, allTagCounts, transitionCounts, emissionCounts, A, B

    perWordTagCounts = {}
    allTagCounts = Counter()
    transitionCounts = {}
    emissionCounts = {}
    for sentence in tagged_sentences:
        prev_tag = START
        for word, tag in sentence:
            insert2dict_of_counters(perWordTagCounts, word, tag)
            insert2dict_of_counters(transitionCounts, prev_tag, tag)
            insert2dict_of_counters(emissionCounts, tag, word)
            allTagCounts[tag] += 1

            prev_tag = tag
        insert2dict_of_counters(transitionCounts, prev_tag, END)

    smooth_dictionary(transitionCounts, set(allTagCounts.keys()) | {END})
    smooth_dictionary(emissionCounts, set(perWordTagCounts.keys()) | {UNK})

    A = get_probabilities_dict(transitionCounts)
    B = get_probabilities_dict(emissionCounts)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    tagged_sentence = []
    for token in sentence:
        if token in perWordTagCounts.keys():
            # get the most frequent tag for this word:
            ctr = perWordTagCounts[token]
            tag = ctr.most_common(1)[0][0]
        else:
            # sample from the tag distribution:
            tag = random.choices(list(allTagCounts.keys()), k=1, weights=list(allTagCounts.values()))[0]
        tagged_sentence.append((token, tag))
    return tagged_sentence

#===========================================
#       POS tagging with HMM
#===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    end_item = viterbi(sentence, A, B)
    tags = retrace(end_item)
    tagged_sentence = list(zip(sentence, tags))

    return tagged_sentence


class ViterbiCell:
    def __init__(self, t, r, p):
        """ Represents a Viterbi cell in the Viterbi matrix.

        :param t: the current tag
        :type t: str
        :param r: the reference to the parent cell in the optimal path to this cell
        :type r: ViterbiCell
        :param p: the log-probability of reaching this cell via the optimal path.
        :type p: float
        """
        self.tag = t
        self.ref = r
        self.prob = p


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tuple (t,r,p), where
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far.

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

    """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    prev_layer = []
    # the start log prob is already initialized to 0 (log-prob of 1)
    v_start = ViterbiCell(t=START, r=None, p=0)
    prev_layer.append(v_start)
    # start going over the given sentence:
    for i, token in enumerate(sentence):
        curr_layer = []
        for tag in A.keys():
            if tag in [START, END]:
                continue
            if token not in B[tag].keys():
                token = UNK
            emission_prob = B[tag][token]
            v_curr = get_best_viterbi_cell(tag, emission_prob, prev_layer, A)
            curr_layer.append(v_curr)
        prev_layer = curr_layer
    v_last = get_best_viterbi_cell(tag=END, emission_prob=0, prev_layer=prev_layer, A=A)

    return v_last


def get_best_viterbi_cell(tag, emission_prob, prev_layer, A):
    """ Finds the best path to the current tag, using the previous layer of ViterbiCells.

    :param tag: the current tag
    :type tag: str
    :param emission_prob: the emission log-probability of the current observation (word) from the current tag
    :type emission_prob: float
    :param prev_layer: the previous layer of ViterbiCells
    :type prev_layer: list
    :param A: the transition probabilities dictionary
    :type A: dict
    :return: the best ViterbiCell for the current tag using the previous layer of ViterbiCells
    :rtype: ViterbiCell
    """
    max_prob, best_prev_tag = None, None
    for prev_tag in prev_layer:
        # prev_tag is a ViterbiCell object
        trans_prob = A[prev_tag.tag][tag]
        prev_prob = prev_tag.prob
        curr_prob = emission_prob + trans_prob + prev_prob
        if max_prob is None or max_prob < curr_prob:
            max_prob = curr_prob
            best_prev_tag = prev_tag
    v_curr = ViterbiCell(t=tag, r=best_prev_tag, p=max_prob)
    return v_curr


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    rev_list = []
    curr_item = end_item.ref  # skipping the END tag.
    while curr_item.ref is not None:  # skipping the START tag.
        rev_list.append(curr_item.tag)
        curr_item = curr_item.ref
    return reversed(rev_list)


# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tuple)
    """
    pass


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): The HMM emission probabilities.
    """
    p = 0   # joint log prob. of words and tags

    prev_tag = START
    for word, tag in sentence:
        trans_p = A[prev_tag][tag]
        if word not in B[tag].keys():
            word = UNK
        emission_p = B[tag][word]
        p += trans_p + emission_p  # log probs are added and not multiplied.
        prev_tag = tag
    # add the prob to get to end:
    trans_p = A[prev_tag][END]
    p += trans_p

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


#===========================================
#       POS tagging with BiLSTM
#===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanila biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""

# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU, LeRU)


class BiLSTMModel(nn.Module):
    def __init__(self, input_dimension, pretrained_embedding_weights, word2idx, embedding_dimension,
                 num_of_layers, tag2idx, idx2tag, output_dimension, input_rep):
        """ Initializes a new BiLSTM model for pos-tagging.
        This model is the same for both Vanilla and Case-Based input representations - the difference is only in the
        amount of features that goes into the LSTM layer.

        :param input_dimension: the maximal sentence length to be input into the LSTM layer
        :type input_dimension: int
        :param pretrained_embedding_weights: the weights of the pretrained embeddings
        :type pretrained_embedding_weights: torch.Tensor
        :param word2idx: a dictionary mapping words to their corresponding index
        :type word2idx: dict
        :param embedding_dimension: the dimension of the embeddings
        :type embedding_dimension: int
        :param num_of_layers: the number of layers that the LSTM layer will stack
        :type num_of_layers: int
        :param tag2idx: a dictionary mapping tags to their corresponding index
        :type tag2idx: dict
        :param idx2tag: a dictionary mapping indexes back to their tags
        :type idx2tag: dict
        :param output_dimension: the amount of tags to be possible candidates in the output layer
        :type output_dimension: int
        :param input_rep: 0 for the vanilla and 1 for the case-base
        :type input_rep: int
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.embedding_dimension = embedding_dimension
        self.num_of_layers = num_of_layers
        self.output_dimension = output_dimension
        self.input_rep = input_rep

        # we will initialize the followings with the pretrained embeddings in the train_rnn function:
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.word2idx = word2idx
        self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding_weights)

        # if we are using the case-based BiLSTM, we need 3 more inputs:
        self.features_num = embedding_dimension + 3 * self.input_rep
        self.lstm_output_dim = 256
        self.dropout_percentage = 0.25 if num_of_layers > 1 else 0
        self.bi_lstm_layer = nn.LSTM(self.features_num,
                                     self.lstm_output_dim,
                                     num_layers=num_of_layers,
                                     dropout=self.dropout_percentage,
                                     batch_first=True, bidirectional=True)

        self.output_layer = nn.Linear(self.lstm_output_dim * 2, output_dimension)
        self.dropout_layer = nn.Dropout(0.25)

    def forward(self, sentence_idx, word_features):
        embedded = self.dropout_layer(self.embedding_layer(sentence_idx))

        if self.input_rep == 0:
            x = embedded
        else:  # self.input_rep == 1
            x = torch.cat((embedded, word_features), dim=2)

        x, _ = self.bi_lstm_layer(x)

        x = self.dropout_layer(x)
        x = self.output_layer(x)

        return x


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       the returned dict may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurrence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimension.
                            If it's value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occurring more that min_frequency are considered.
                        min_frequency provides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should specify default values.
    Return:
        a dictionary with at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """
    max_vocab_size = params_d['max_vocab_size']
    min_frequency = params_d['min_frequency']
    pretrained_embeddings_fn = params_d['pretrained_embeddings_fn']
    data_fn = params_d['data_fn']

    # preprocess the embeddings:
    sentences = load_annotated_corpus(data_fn)
    # count unique words appearances in data:
    c = Counter(word.lower() for sentence in sentences for word, tag in sentence)
    # if max_vocab_size == -1, do not limit the vocab size:
    most_common_words = c.most_common(max_vocab_size if max_vocab_size != -1 else None)
    vocab = []
    for word, freq in most_common_words:
        if freq >= min_frequency:
            vocab.append(word)

    vectors = load_pretrained_embeddings(pretrained_embeddings_fn, vocab)

    # calculate the input_dimension (longest sentence in the training file):
    input_dimension = max(len(sentence) for sentence in sentences) + 1  # +1 for the special EOS_WORD

    # create tag2idx and idx2tag:
    possible_tags = {tag for sentence in sentences for word, tag in sentence} | {EOS_TAG}
    output_dimension = params_d['output_dimension'] + 1  # +1 for the EOS_TAG
    if len(possible_tags) > output_dimension:
        raise Exception('The output_dimension must be at least the amount of tags in the training set.')
    tag2idx = {}
    idx2tag = {}
    for i, tag in enumerate(possible_tags):
        tag2idx[tag] = i
        idx2tag[i] = tag

    # get remaining parameters from the params_d:
    num_of_layers = params_d['num_of_layers']
    input_rep = params_d['input_rep']
    embedding_dimension = params_d['embedding_dimension']

    model = BiLSTMModel(input_dimension=input_dimension,
                        pretrained_embedding_weights=vectors['weights'],
                        word2idx=vectors['word2idx'],
                        embedding_dimension=embedding_dimension,
                        num_of_layers=num_of_layers,
                        tag2idx=tag2idx,
                        idx2tag=idx2tag,
                        output_dimension=output_dimension,
                        input_rep=input_rep)

    # we do not need to add the embeddings and vocab, because it is inside of the model parameters:
    model_d = {'lstm': model,
               'input_rep': input_rep}

    return model_d


# no need for this one as part of the API
'''
def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        output_dimension': int}
    """

    params_d = {'input_dimension': model.input_dimension,
                'embedding_dimension': model.embedding_dimension,
                'num_of_layers': model.num_of_layers,
                'output_dimension': model.output_dimension,
                'vanila': model.vanila}

    return params_d
'''

# end of sentence:
EOS_WORD = '<DUMMY_EOS>'
EOS_TAG = END


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    if vocab:
        limiting_embedding_words = True
        v_set = {word.lower() for word in vocab}
    else:
        limiting_embedding_words = False

    embeddings_dict = {}
    embedding_dim = None
    with open(path, mode="r", encoding='UTF-8') as file1:
        all_lines = file1.readlines()
        for line in all_lines:
            split_line = line.strip().split()
            word = split_line[0].lower()
            if (not limiting_embedding_words) or (word in v_set):
                embeddings_dict[word] = np.array(split_line[1:]).astype(np.float32)
                if embedding_dim is None:
                    embedding_dim = len(embeddings_dict[word])

    vocab = set(embeddings_dict.keys())

    word2idx = {}
    weights = []
    for i, word in enumerate(vocab):
        word2idx[word] = i
        weights.append(embeddings_dict[word])

    if limiting_embedding_words:
        # add the remaining words that are in v_set but not in vocab:
        remaining = v_set - vocab
        for word in remaining:
            add_word_to_embeddings(word2idx, weights, word)

    # add special words:
    add_word_to_embeddings(word2idx, weights, UNK)
    add_word_to_embeddings(word2idx, weights, EOS_WORD)

    weights = torch.Tensor(weights)

    vectors = {'word2idx': word2idx,
               'weights': weights}

    return vectors


def preprocess_word(word, word2idx, input_rep):
    """ Creates features for the given word, according to the input_rep.

    :param word: the word to create features for
    :type word: str
    :param word2idx: a dictionary mapping each word to it's corresponding index for the embeddings usage
    :type word2idx: dict
    :param input_rep: 0 for the vanilla and 1 for the case-base
    :type input_rep: int
    :return: the features for the current word, in the format: (embedding_index, case_features) - where case_features
        will an empty list for vanilla and the 3-case-features for the case-base
    :rtype: tuple
    """
    if input_rep == 0:
        case_features = []
    else:  # input_rep == 1
        case_features = [word.islower(), word.isupper(), word[0].isupper()]
    if word == EOS_WORD:  # special case - the EOS_WORD will be indexed with upper-case and not lower-case
        idx = word2idx[word]
    else:
        lowered_word = word.lower()
        idx = word2idx[lowered_word] if lowered_word in word2idx.keys() else word2idx[UNK]
    return idx, case_features


def preprocess_sentence(sentence, word2idx, input_rep):
    """ Creates features for the entire sentence. The features will be a list in the length of the sentence.

    :param sentence: the sentence to create features for
    :type sentence: list
    :param word2idx: a dictionary mapping each word to it's corresponding index for the embeddings usage
    :type word2idx: dict
    :param input_rep: 0 for the vanilla and 1 for the case-base
    :type input_rep: int
    :return: the features for the given sentence, in the format: (embedding_indices, case_features) - both lists of
        features returned by the preprocess_word function
    :rtype: tuple
    """
    idx_sentence = []
    case_sentence = []
    for word in sentence:
        idx, case = preprocess_word(word=word, word2idx=word2idx, input_rep=input_rep)
        idx_sentence.append(idx)
        case_sentence.append(case)

    return idx_sentence, case_sentence


def add_word_to_embeddings(word2idx, weights, word):
    """ Adds a new word to the pretrained embeddings. Generates random weights for the new word, the weights shall be
        learned in the training process.

    :param word2idx: a dictionary mapping each word to it's corresponding index for the embeddings usage
    :type word2idx: dict
    :param weights: a list where each word's index has it's pretrained embeddings
    :type weights: list
    :param word: the word to add to the pretrained embeddings
    :type word: str
    """
    idx = len(word2idx.keys())
    embedding_dim = len(weights[0])
    word2idx[word] = idx
    weight = np.random.randn(embedding_dim)  # random nd array, with values from the standard normal distribution
    weights.append(weight)


def preprocess_data(train_data, word2idx, tag2idx, sentence_length, input_rep):
    """ Creates features for the entire train data.

    :param train_data: the data to create features for
    :type train_data: list
    :param word2idx: a dictionary mapping each word to it's corresponding index for the embeddings usage
    :type word2idx: dict
    :param tag2idx: a dictionary mapping tags to their corresponding index
    :type tag2idx: dict
    :param sentence_length: the maximal sentence length to be input into the LSTM layer
    :type sentence_length: int
    :param input_rep: 0 for the vanilla and 1 for the case-base
    :type input_rep: int
    :return: a tuple of the features for the training data, and the tags for each sentence, in the format of:
        (X_indices, X_case, y) - where all of them are tensors, X_indices contains the indices of each sentence in the
        training data, X_case contains the case vectors for each sentence, and y contains the tags indices for each
        sentence.
    :rtype: tuple
    """
    X_idx_sentences = []
    X_case_sentences = []
    y_sentences = []
    for sentence in train_data:
        words = []
        tags = []
        for word, tag in sentence:
            words.append(word)
            if tag not in tag2idx.keys():
                raise Exception('The tags in the corpus does not match the tags given in the initialize model function')
            idx = tag2idx[tag]
            tags.append(idx)
        # pad the sentence:
        diff = sentence_length - len(sentence)
        if diff <= 0:
            raise Exception(f'Need to increase the input_dim by at least {-diff + 1}')
        words += diff * [EOS_WORD]
        tags += diff * [tag2idx[EOS_TAG]]

        X_idx_sentence, X_case_sentence = preprocess_sentence(sentence=words, word2idx=word2idx, input_rep=input_rep)

        X_idx_sentences.append(X_idx_sentence)
        X_case_sentences.append(X_case_sentence)
        y_sentences.append(tags)
    X_idx = torch.LongTensor(X_idx_sentences)
    X_case = torch.BoolTensor(X_case_sentences)
    y = torch.LongTensor(y_sentences)

    return X_idx, X_case, y


def get_batches(X_idx, X_case, y, batch_size):
    """ A generator for random batches of data.

    :param X_idx: the indices of all of the sentences in the training data
    :type X_idx: torch.LongTensor
    :param X_case: the case features of all of the sentences in the training data
    :type X_case: torch.BoolTensor
    :param y: the tags indices of all of the sentences in the training data
    :type y: torch.LongTensor
    :param batch_size: the size of each batch of data to yield at every iteration
    :type batch_size: int
    :return: at each iteration, yields the current batch of features and their corresponding tags, in the format:
        (batch_X_idx, batch_X_case, batch_y) - X_idx = indices features, X_case = case features, and y = tags indices.
        All of them are tensors.
    :rtype: tuple
    """
    idx_list = np.arange(len(X_idx))
    np.random.shuffle(idx_list)
    i = 0
    while i * batch_size < len(idx_list):
        indices = idx_list[i * batch_size:(i + 1) * batch_size]

        batch_X_idx = X_idx[indices]
        batch_X_case = X_case[indices]
        batch_y = y[indices]
        yield batch_X_idx, batch_X_case, batch_y
        i += 1


# The next class is an EarlyStopping callback that was taken from:
# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)
    lstm_model = model['lstm']
    input_rep = model['input_rep']

    criterion = nn.CrossEntropyLoss()  # you can set the parameters as you like

    word2idx = lstm_model.word2idx
    tag2idx = lstm_model.tag2idx
    sentence_length = lstm_model.input_dimension

    X_idx, X_case, y = preprocess_data(train_data=train_data,
                                       word2idx=word2idx,
                                       tag2idx=tag2idx,
                                       sentence_length=sentence_length,
                                       input_rep=input_rep)

    if val_data:
        val_X_idx, val_X_case, val_y = preprocess_data(train_data=val_data,
                                                       word2idx=word2idx,
                                                       tag2idx=tag2idx,
                                                       sentence_length=sentence_length,
                                                       input_rep=input_rep)
    else:
        # take 80% of the train data to be validation and it will determine the early stopping:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        val_indices = indices[:int(0.2 * len(indices))]
        train_indices = indices[int(0.2 * len(indices)):]

        val_X_idx, val_X_case, val_y = X_idx[val_indices], X_case[val_indices], y[val_indices]
        X_idx, X_case, y = X_idx[train_indices], X_case[train_indices], y[train_indices]

    val_X_idx, val_X_case, val_y = val_X_idx.to(device), val_X_case.to(device), val_y.to(device)
    val_y = val_y.reshape(-1)

    optimizer = Adam(lstm_model.parameters(), lr=0.005)  # lr optimized in our experiments to 0.005
    epochs = 100
    batch_size = 128

    lstm_model = lstm_model.to(device)
    criterion = criterion.to(device)

    # initialize the early_stopping object
    # will early stop if the validation set loss will not improve in a range of 5 epochs:
    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in range(epochs):
        lstm_model.train()  # prep model for training
        for batch_X_idx, batch_X_case, batch_y in get_batches(X_idx, X_case, y, batch_size):
            batch_X_idx = batch_X_idx.to(device)
            batch_X_case = batch_X_case.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            y_pred = lstm_model(batch_X_idx, batch_X_case)
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            batch_y = batch_y.reshape(-1)

            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

        # Checking loss on validation set:
        lstm_model.eval()  # prep model for evaluation
        # forward pass: compute predicted outputs by passing inputs to the model
        y_pred = lstm_model(val_X_idx, val_X_case)
        # calculate the loss
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        valid_loss = criterion(y_pred, val_y)

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, lstm_model)

        if early_stopping.early_stop:
            break

    # load the last checkpoint with the best model
    lstm_model.load_state_dict(torch.load('checkpoint.pt'))


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    lstm_model = model['lstm']
    input_rep = model['input_rep']

    # pad the sentence:
    words = sentence + [EOS_WORD]
    X_idx, X_case = preprocess_sentence(sentence=words, word2idx=lstm_model.word2idx, input_rep=input_rep)
    X_idx_tensor = torch.LongTensor(X_idx).reshape(1, -1).to(device)
    X_case_tensor = torch.BoolTensor(X_case).reshape(1, -1, 3).to(device)

    predictions = lstm_model(X_idx_tensor, X_case_tensor)
    tags_idx = torch.argmax(predictions, dim=-1).reshape(-1)
    tags = [lstm_model.idx2tag[int(idx)] if int(idx) in lstm_model.idx2tag.keys() else UNK for idx in tags_idx]
    tags = tags[:len(sentence)]

    tagged_sentence = list(zip(sentence, tags))

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a dictionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    model_params = {'max_vocab_size': -1,
                    'min_frequency': 2,
                    'input_rep': 1,
                    'embedding_dimension': 100,
                    'num_of_layers': 3,
                    'output_dimension': 17,
                    'pretrained_embeddings_fn': 'glove.6B.100d.txt',
                    'data_fn': 'en-ud-train.upos.tsv'
                    }

    return model_params


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model itself and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM transition probabilities
        B (dict): The HMM emission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, model['baseline'][0], model['baseline'][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, model['hmm'][0], model['hmm'][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, model['blstm'][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, model['cblstm'][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    global perWordTagCounts

    assert len(gold_sentence) == len(pred_sentence)
    correct, correctOOV, OOV = 0, 0, 0
    for gold, pred in zip(gold_sentence, pred_sentence):
        assert gold[0] == pred[0]
        word = gold[0]
        if word in perWordTagCounts.keys():
            # in the vocabulary
            if gold[1] == pred[1]:
                correct += 1
        else:
            # OOV
            OOV += 1
            if gold[1] == pred[1]:
                correctOOV += 1
                correct += 1

    return correct, correctOOV, OOV
