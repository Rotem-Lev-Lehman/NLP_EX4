import tagger

train_path = 'en-ud-train.upos.tsv'
dev_path = 'en-ud-dev.upos.tsv'

train_data = tagger.load_annotated_corpus(train_path)
dev_data = tagger.load_annotated_corpus(dev_path)

[allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B] = tagger.learn_params(train_data)

total_correct, total_correctOOV, total_OOV, total_length = 0, 0, 0, 0
#for gold_sentence in dev_data:
gold_sentence = train_data[0]
pred_sentence = [w[0] for w in gold_sentence]

#tagged_sentence = tagger.baseline_tag_sentence(pred_sentence, perWordTagCounts, allTagCounts)
tagged_sentence = tagger.hmm_tag_sentence(pred_sentence, A, B)
correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)
total_correct += correct
total_correctOOV += correctOOV
total_OOV += OOV
total_length += len(gold_sentence) - OOV

prob1 = tagger.joint_prob(gold_sentence, A, B)
prob2 = tagger.joint_prob(tagged_sentence, A, B)

print(gold_sentence)
print(prob1)
print(tagged_sentence)
print(prob2)
print(f"correct: {total_correct}, non_OOV: {total_length}, correctOOV: {total_correctOOV}, OOV: {total_OOV}")
