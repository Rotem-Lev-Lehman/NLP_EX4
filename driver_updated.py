
from tagger import tag_sentence, count_correct, load_annotated_corpus, learn_params, initialize_rnn_model, train_rnn, joint_prob, get_best_performing_model_params
from random import randrange


def calc_score(dev_data, model_dict):
    score_nom, score_denom = 0, 0
    for gold_sentence in dev_data:
        pred_sentence = [w[0] for w in gold_sentence]
        tagged_sentence = tag_sentence(pred_sentence, model_dict)
        correct, correctOOV, OOV = count_correct(gold_sentence, tagged_sentence)
        score_nom += correct
        score_denom += len(pred_sentence)
    print(f"{list(model_dict.keys())[0]} score is {score_nom / score_denom}")


def check_sampled_sentence(gold_sentence, model_dict):
    pred_sentence = [w[0] for w in gold_sentence]
    tagged_sentence = tag_sentence(pred_sentence, model_dict)
    correct, correctOOV, OOV = count_correct(gold_sentence, tagged_sentence)
    print(f"correct: {correct}, correctOOV: {correctOOV}, OOV: {OOV}\n")


train_path = r"en-ud-train.upos.tsv"
dev_path = r"en-ud-dev.upos.tsv"

train_data = load_annotated_corpus(train_path)
dev_data = load_annotated_corpus(dev_path)

[allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] = learn_params(train_data)


# draw random sentence
gold_sentence = dev_data[randrange(len(dev_data))]
print(f"tested random sentence is {gold_sentence} of length {len(gold_sentence)}\n")


# test baseline
calc_score(dev_data, {'baseline': [perWordTagCounts, allTagCounts]})
check_sampled_sentence(gold_sentence, {'baseline': [perWordTagCounts, allTagCounts]})


# test hmm
calc_score(dev_data, {'hmm': [A,B]})
check_sampled_sentence(gold_sentence, {'hmm': [A,B]})

# test joint_prob:
p = joint_prob(gold_sentence, A, B)
print(f'joint prob of gold sentence: {p}')

print(f'Amount of different tags = {len(set(B.keys()))}')

# LSTM settings:
pretrained_embeddings_fn = r"glove.6B.100d.txt"
model_dict = {'max_vocab_size': -1,  # max vocabulary size(int)
              'min_frequency': 2,  # the occurrence threshold to consider(int)
              'input_rep': 0,
              'embedding_dimension': 100,  # embedding vectors size (int)
              'num_of_layers': 2,  # number of layers (int)
              'output_dimension': len(set(B.keys())),  # number of tags in tagset (int)
              'pretrained_embeddings_fn': pretrained_embeddings_fn,  # str
              'data_fn': train_path  # str
              }

model_dict = get_best_performing_model_params()
# test Vanilla BiLSTM:
print(f"provided model dict is {model_dict}")
print(f"initializing model")
model = initialize_rnn_model(model_dict)
print(f"training model")
train_rnn(model, train_data)
print(f"evaluating model")
calc_score(dev_data, {'blstm': [model]})
check_sampled_sentence(gold_sentence, {'blstm': [model]})

long_sentence = gold_sentence * 300
print(long_sentence)
print(len(long_sentence))
check_sampled_sentence(gold_sentence, {'blstm': [model]})

exit(0)
# test BiLSTM + case:
model_dict['input_rep'] = 1
print(f"provided model dict is {model_dict}")
print(f"initializing model")
model = initialize_rnn_model(model_dict)
print(f"training model")
train_rnn(model, train_data)
print(f"evaluating model")
calc_score(dev_data, {'cblstm': [model]})
check_sampled_sentence(gold_sentence, {'cblstm': [model]})


