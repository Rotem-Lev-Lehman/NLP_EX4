import tagger

train_path = 'en-ud-train.upos.tsv'
dev_path = 'en-ud-dev.upos.tsv'
embedding_path = 'glove.6B.100d.txt'

params = {'input_dimension': 200,
          'embedding_dimension': 100,
          'num_of_layers': 1,
          'output_dimension': 45,
          'vanila': False}

# tagger.save_embedding_pickle(embedding_path)

model = tagger.initialize_rnn_model(params)
tagger.train_rnn(model, train_path, embedding_path)
tagged_sentence = tagger.rnn_tag_sentence(['Hi', 'whats', 'up', '?'], model)
