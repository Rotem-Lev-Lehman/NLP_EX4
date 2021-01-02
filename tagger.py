"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
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
    #TODO edit the dictionary to have your own details
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
    if key1 not in d.keys():
        d[key1] = Counter()
    d[key1][key2] += 1


def get_probabilities_dict(d):
    prob_dict = {}
    for key1 in d.keys():
        ctr = d[key1]
        total = sum(ctr.values())
        prob_dict[key1] = {key2: log(count / total) for key2, count in ctr.items()}
    return prob_dict


def smooth_dictionary(d, smooth_options):
    for key1 in d.keys():
        ctr = d[key1]
        for key2 in smooth_options:
            ctr[key2] += 1


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    global perWordTagCounts, allTagCounts, transitionCounts, emissionCounts, A, B

    # TODO - check what to do with the UNK (unknown) tag
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
    smooth_dictionary(emissionCounts, perWordTagCounts.keys())

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
        # TODO - Check the hints:
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
            emission_prob = B[tag][token]  # TODO - handle OOV
            v_curr = get_best_viterbi_cell(tag, emission_prob, prev_layer, A)
            curr_layer.append(v_curr)
        prev_layer = curr_layer
    # TODO - check if need the following code (for the probability of END):
    v_last = get_best_viterbi_cell(tag=END, emission_prob=0, prev_layer=prev_layer, A=A)

    return v_last


def get_best_viterbi_cell(tag, emission_prob, prev_layer, A):
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


#a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """

#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tuple)
    """


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
        # TODO - Check what to do with OOV words and tags
        trans_p = A[prev_tag][tag]
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

def initialize_rnn_model(params_d):
    """Returns an lstm model based on the specified parameters.

    Args:
        params_d (dict): an dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'input_dimension': int,
                        'embedding_dimension': int,
                        'num_of_layers': int,
                        'output_dimension': int}
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        torch.nn.Module object
    """

    #TODO complete the code

    return model

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

    #TODO complete the code

    return params_d

def load_pretrained_embeddings(path):
    """ Returns an object with the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the data structure of your choice.
    """
    #TODO
    return vectors


def train_rnn(model, data_fn, pretrained_embeddings_fn):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (torch.nn.Module): the model to train
        data_fn (string): full path to the file with training data (in the provided format)
        pretrained_embeddings_fn (string): full path to the file with pretrained embeddings
    """
    #Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider loading the data and preprocessing it
    # 4. consider using batching
    # 5. some of the above could be implemented in helper functions (not part of
    #    the required API)

    #TODO complete the code

    criterion = nn.CrossEntropyLoss() #you can set the parameters as you like
    vectors = load_pretrained_embeddings(pretrained_embeddings_fn)

    model = model.to(device)
    criterion = criterion.to(device)


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence. Tagging is done with the Viterby
        algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (torch.nn.Module):  a trained BiLSTM model

    Return:
        list: list of pairs
    """

    #TODO complete the code

    return tagged_sentence

def get_best_performing_model_params():
    """Returns a dictionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    #TODO complete the code

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
        or the model itself (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[Torch.nn.Module]}
        4. BiLSTM+case: {'cblstm': [Torch.nn.Module]}
        5. (NOT REQUIRED: you can add other variations, augmenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the LSTM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, model.values()[0], model.values()[1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, model.values()[0], model.values()[1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, model.values()[0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, model.values()[0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
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

    return correct, correctOOV, OOV
