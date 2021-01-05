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
    rev_list = []
    curr_item = end_item.ref  # skipping the END tag.
    while curr_item.ref is not None:  # skipping the START tag.
        rev_list.append(curr_item.tag)
        curr_item = curr_item.ref
    return reversed(rev_list)


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


class BiLSTMModel(nn.Module):
    def __init__(self, input_dimension, embedding_dimension, num_of_layers, output_dimension, vanila=True):
        super().__init__()
        self.input_dimension = input_dimension
        self.embedding_dimension = embedding_dimension
        self.num_of_layers = num_of_layers
        self.output_dimension = output_dimension
        self.vanila = vanila

        # we will initialize the followings with the pretrained embeddings in the train_rnn function:
        self.tag2idx = None
        self.idx2tag = None
        self.word2idx = None
        self.embedding_layer = None

        # if we are using the case-based BiLSTM, we need 3 more inputs:
        self.features_num = embedding_dimension + 3 * int(not vanila)
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

        if self.vanila:
            x = embedded
        else:
            x = torch.cat((embedded, word_features), dim=1)

        x, _ = self.bi_lstm_layer(x)

        x = self.dropout_layer(x)
        x = self.output_layer(x)

        return x


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

    model = BiLSTMModel(**params_d)

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

    params_d = {'input_dimension': model.input_dimension,
                'embedding_dimension': model.embedding_dimension,
                'num_of_layers': model.num_of_layers,
                'output_dimension': model.output_dimension,
                'vanila': model.vanila}

    return params_d


# end of sentence:
EOS_WORD = '<DUMMY_EOS>'
EOS_TAG = END


def save_embedding_pickle(path):
    import pickle
    vectors = load_pretrained_embeddings(path)
    with open('vectors.pickle', 'wb') as f:
        pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings_pickle():
    import pickle
    with open('vectors.pickle', 'rb') as f:
        vectors = pickle.load(f)
    return vectors


def load_pretrained_embeddings(path):
    """ Returns an object with the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the data structure of your choice.
    """
    embeddings_dict = {}
    embedding_dim = None
    with open(path, mode="r", encoding='UTF-8') as file1:
        all_lines = file1.readlines()
        for line in all_lines:
            split_line = line.strip().split()
            embeddings_dict[split_line[0]] = np.array(split_line[1:]).astype(np.float32)
            if embedding_dim is None:
                embedding_dim = len(embeddings_dict[split_line[0]])

    print('Done embeddings')
    # add special words:
    embeddings_dict[UNK] = np.zeros(embedding_dim)
    embeddings_dict[EOS_WORD] = np.zeros(embedding_dim)

    vocab = set(embeddings_dict.keys())

    word2idx = {}
    weights = []
    for i, word in enumerate(vocab):
        word2idx[word] = i
        weights.append(embeddings_dict[word])

    pretrained_weights = torch.Tensor(weights)

    vectors = {'word2idx': word2idx,
               'weights': pretrained_weights}

    return vectors


def preprocess_word(word, word2idx, vanila=True):
    if vanila:
        case_features = []
    else:
        case_features = [word.islower(), word.isupper(), word[0].isupper()]
    idx = word2idx[word] if word in word2idx.keys() else word2idx[UNK]
    return idx, case_features


def preprocess_sentence(sentence, word2idx, vanila=True):
    idx_sentence = []
    case_sentence = []
    for word in sentence:
        idx, case = preprocess_word(word, word2idx, vanila)
        idx_sentence.append(idx)
        case_sentence.append(case)

    return idx_sentence, case_sentence


def preprocess_data(data_fn, word2idx, sentence_length, vanila=True):
    sentences = load_annotated_corpus(data_fn)
    tag2idx = {EOS_TAG: 0}
    next_tag_index = 1

    X_idx_sentences = []
    X_case_sentences = []
    y_sentences = []
    print('preprocessing data')
    for sentence in sentences:
        words = []
        tags = []
        for word, tag in sentence:
            words.append(word)

            if tag not in tag2idx.keys():
                tag2idx[tag] = next_tag_index
                next_tag_index += 1
            idx = tag2idx[tag]

            tags.append(idx)
        # pad the sentence:
        diff = sentence_length - len(sentence)
        if diff <= 0:
            raise Exception(f'Need to increase the input_dim by at least {-diff + 1}')
        words += diff * [EOS_WORD]

        tags += diff * [tag2idx[EOS_TAG]]

        X_idx_sentence, X_case_sentence = preprocess_sentence(words, word2idx, vanila)

        X_idx_sentences.append(X_idx_sentence)
        X_case_sentences.append(X_case_sentence)
        y_sentences.append(tags)
    print('done preprocess')
    X_idx = torch.LongTensor(X_idx_sentences)
    X_case = torch.BoolTensor(X_case_sentences)
    y = torch.LongTensor(y_sentences)

    idx2tag = {val: key for key, val in tag2idx.items()}

    return X_idx, X_case, y, tag2idx, idx2tag


def get_batches(X_idx, X_case, y, batch_size):
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

    criterion = nn.CrossEntropyLoss()  # you can set the parameters as you like
    # vectors = load_pretrained_embeddings(pretrained_embeddings_fn)
    vectors = load_embeddings_pickle()
    model.word2idx = vectors['word2idx']
    model.embedding_layer = nn.Embedding.from_pretrained(vectors['weights'])

    X_idx, X_case, y, tag2idx, idx2tag = preprocess_data(data_fn, word2idx=model.word2idx, sentence_length=model.input_dimension, vanila=model.vanila)
    model.tag2idx = tag2idx
    model.idx2tag = idx2tag

    optimizer = Adam(model.parameters())  # TODO - change lr?
    epochs = 100  # TODO - change this
    batch_size = 128  # TODO - check this

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        print(f'Epoch num {epoch}')
        for i, (batch_X_idx, batch_X_case, batch_y) in enumerate(get_batches(X_idx, X_case, y, batch_size)):
            print(f'Batch number {i}')
            batch_X_idx = batch_X_idx.to(device)
            batch_X_case = batch_X_case.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            y_pred = model(batch_X_idx, batch_X_case)
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            batch_y = batch_y.reshape(-1)

            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()


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
    # pad the sentence:
    sentence_length = model.input_dimension
    diff = sentence_length - len(sentence)
    if diff <= 0:
        raise Exception(f'Need to increase the input_dim by at least {-diff + 1}')
    words = sentence + diff * [EOS_WORD]
    X_idx, X_case = preprocess_sentence(words, word2idx=model.word2idx, vanila=model.vanila)
    X_idx_tensor = torch.LongTensor(X_idx).reshape(1, sentence_length).to(device)
    X_case_tensor = torch.BoolTensor(X_idx).reshape(1, sentence_length).to(device)

    predictions = model(X_idx_tensor, X_case_tensor)
    tags_idx = torch.argmax(predictions, dim=-1).reshape(-1)
    tags = [model.idx2tag[int(idx)] if int(idx) in model.idx2tag.keys() else UNK for idx in tags_idx]
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
