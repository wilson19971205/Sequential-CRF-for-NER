# models.py

from numpy.core.fromnumeric import sort
from numpy.lib.function_base import gradient
from optimizers import *
from nerdata import *
from utils import *

import random
import time

from collections import Counter
from typing import List

import numpy as np


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        score = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        back_pointers = np.ones((len(sentence_tokens), len(self.tag_indexer))) * -1
        sequence_scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
        for tag_idx in range(0, len(self.tag_indexer)):    
            score[0][tag_idx] = sequence_scorer.score_init(sentence_tokens, tag_idx) + sequence_scorer.score_emission(sentence_tokens, tag_idx, 0)

        for idx in range(1, len(sentence_tokens)):
            for curr_tag_idx in range(0, len(self.tag_indexer)):
                score[idx][curr_tag_idx] = -np.inf
                for prev_tag_idx in range(0, len(self.tag_indexer)):
                    curr_score = sequence_scorer.score_transition(sentence_tokens, prev_tag_idx, curr_tag_idx) + sequence_scorer.score_emission(sentence_tokens, curr_tag_idx, idx) + score[idx-1][prev_tag_idx]
                    if curr_score > score[idx][curr_tag_idx]:
                        score[idx][curr_tag_idx] = curr_score
                        back_pointers[idx][curr_tag_idx] = prev_tag_idx

        idx = np.argmax(score,axis=1)[-1]
        pred_tags = []
        count = len(sentence_tokens) - 1
        while idx != -1 :
            pred_tags.append(self.tag_indexer.get_object(idx))
            idx = int(back_pointers[count][idx])
            count = count - 1
        pred_tags.reverse()

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_hmm_model(sentences: List[LabeledSentence], silent: bool=False) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    if not silent:
        print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    if not silent:
        print("Tag indexer: %s" % tag_indexer)
        print("Initial state log probabilities: %s" % init_counts)
        print("Transition log probabilities: %s" % transition_counts)
        print("Emission log probs too big to print...")
        print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
        print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
        print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


##################
# CRF code follows

class FeatureBasedSequenceScorer(object):
    """
    Feature-based sequence scoring model. Note that this scorer is instantiated *for every example*: it contains
    the feature cache used for that example.
    """
    def __init__(self, tag_indexer, feature_weights, feat_cache):
        self.tag_indexer = tag_indexer
        self.feature_weights = feature_weights
        self.feat_cache = feat_cache

    def score_init(self, sentence, tag_idx):
        if isI(self.tag_indexer.get_object(tag_idx)):
            return -1000
        else:
            return 0

    def score_transition(self, sentence_tokens, prev_tag_idx, curr_tag_idx):
        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
        if (isO(prev_tag) and isI(curr_tag))\
                or (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) \
                or (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
            return -1000
        else:
            return 0

    def score_emission(self, sentence_tokens, tag_idx, word_posn):
        feats = self.feat_cache[word_posn][tag_idx]
        return self.feature_weights.score(feats)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        #raise Exception("IMPLEMENT ME")
        feature_cache = [[[] for k in range(0, len(self.tag_indexer))] for j in range(0, len(sentence_tokens))]
        for word_idx in range(0, len(sentence_tokens)):
            for tag_idx in range(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)

        # Viterbi
        score = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        back_pointers = np.ones((len(sentence_tokens), len(self.tag_indexer))) * -1
        sequence_scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_weights, feature_cache)
        
        for word_idx in range(0, len(sentence_tokens)):
            if word_idx == 0:
                for tag_idx in range(0, len(self.tag_indexer)):
                    tag = self.tag_indexer.get_object(tag_idx)
                    if isI(tag):
                        score[word_idx][tag_idx] = -np.inf
                    else:    
                        score[word_idx][tag_idx] = sequence_scorer.score_init(feature_cache, tag_idx)
            else:
                for curr_tag_idx in range(0, len(self.tag_indexer)):
                    score[word_idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in range(0, len(self.tag_indexer)):
                        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
                        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                        if isO(prev_tag) and isI(curr_tag):
                            continue
                        if isI(curr_tag) and (get_tag_label(curr_tag) != get_tag_label(prev_tag)):
                            continue
                        curr_score = sequence_scorer.score_transition(feature_cache, prev_tag_idx, curr_tag_idx) + \
                                        sequence_scorer.score_emission(feature_cache, curr_tag_idx, word_idx) + score[word_idx-1][prev_tag_idx]
                        if curr_score > score[word_idx][curr_tag_idx]:
                            score[word_idx][curr_tag_idx] = curr_score
                            back_pointers[word_idx][curr_tag_idx] = prev_tag_idx
        
        max_score_idx = np.argmax(score,axis=1)[-1]
        idx = max_score_idx
        pred_tags = []
        word_idx = len(sentence_tokens) - 1
        while idx != -1 :
            pred_tags.append(self.tag_indexer.get_object(idx))
            idx = int(back_pointers[word_idx][idx])
            word_idx -= 1
        pred_tags.reverse()

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))

    def decode_beam(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        beam_search_k = 4

        feature_cache = [[[] for k in range(0, len(self.tag_indexer))] for j in range(0, len(sentence_tokens))]
        for word_idx in range(0, len(sentence_tokens)):
            for tag_idx in range(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)

        score = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        back_pointers = np.ones((len(sentence_tokens), len(self.tag_indexer))) * -1
        sequence_scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_weights, feature_cache)
        
        
        beam_search_buffer = np.zeros(beam_search_k)
        for word_idx in range(1, len(sentence_tokens)):
            if word_idx == 0:
                for tag_idx in range(0, len(self.tag_indexer)):
                    tag = self.tag_indexer.get_object(tag_idx)
                    if isI(tag):
                        score[word_idx][tag_idx] = -np.inf
                    else:    
                        score[word_idx][tag_idx] = sequence_scorer.score_init(feature_cache, tag_idx)
            elif word_idx == 1:
                for curr_tag_idx in range(0, len(self.tag_indexer)):
                    score[word_idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in range(0, beam_search_k):
                        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
                        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                        if isO(prev_tag) and isI(curr_tag):
                            continue
                        if isI(curr_tag) and (get_tag_label(curr_tag) != get_tag_label(prev_tag)):
                            continue
                        curr_score = sequence_scorer.score_transition(feature_cache, prev_tag_idx, curr_tag_idx) + \
                                        sequence_scorer.score_emission(feature_cache, curr_tag_idx, word_idx) + score[word_idx-1][prev_tag_idx]
                        if curr_score > score[word_idx][curr_tag_idx]:
                            score[word_idx][curr_tag_idx] = curr_score
                            back_pointers[word_idx][curr_tag_idx] = prev_tag_idx
                beam_search_buffer = score[word_idx].argsort()[::-1][0:beam_search_k]
            else:
                for curr_tag_idx in range(0, len(self.tag_indexer)):
                    score[word_idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in range(0, len(beam_search_buffer)):
                        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
                        prev_tag = self.tag_indexer.get_object(beam_search_buffer[prev_tag_idx])
                        if isO(prev_tag) and isI(curr_tag):
                            continue
                        if isI(curr_tag) and (get_tag_label(curr_tag) != get_tag_label(prev_tag)):
                            continue
                        curr_score = sequence_scorer.score_transition(feature_cache, beam_search_buffer[prev_tag_idx], curr_tag_idx) + \
                                        sequence_scorer.score_emission(feature_cache, curr_tag_idx, word_idx) + score[word_idx-1][beam_search_buffer[prev_tag_idx]]
                        if curr_score > score[word_idx][curr_tag_idx]:
                            score[word_idx][curr_tag_idx] = curr_score
                            back_pointers[word_idx][curr_tag_idx] = beam_search_buffer[prev_tag_idx]
                beam_search_buffer = score[word_idx].argsort()[::-1][0:beam_search_k]

        max_score_idx = np.argmax(score,axis=1)[-1]
        idx = max_score_idx
        pred_tags = []
        word_idx = len(sentence_tokens) - 1
        while idx != -1 :
            pred_tags.append(self.tag_indexer.get_object(idx))
            idx = int(back_pointers[word_idx][idx])
            word_idx -= 1
        pred_tags.reverse()

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_crf_model(sentences: List[LabeledSentence], silent: bool=False) -> CrfNerModel:
    """
    Trains a CRF NER model on the given corpus of sentences.
    :param sentences: The training data
    :param silent: True to suppress output, false to print certain debugging outputs
    :return: The CrfNerModel, which is primarily a wrapper around the tag + feature indexers as well as weights
    """
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    if not silent:
        print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0 and not silent:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    if not silent:
        print("Training")
    weight_vector = UnregularizedAdagradTrainer(np.zeros((len(feature_indexer))), eta=1.0)
    num_epochs = 2
    random.seed(0)
    for epoch in range(0, num_epochs):
        epoch_start = time.time()
        if not silent:
            print("Epoch %i" % epoch)
        sent_indices = [i for i in range(0, len(sentences))]
        random.shuffle(sent_indices)
        total_obj = 0.0
        for counter, i in enumerate(sent_indices):
            if counter % 100 == 0 and not silent:
                print("Ex %i/%i" % (counter, len(sentences)))
            scorer = FeatureBasedSequenceScorer(tag_indexer, weight_vector, feature_cache[i])
            (gold_log_prob, gradient) = compute_gradient(sentences[i], tag_indexer, scorer, feature_indexer)
            total_obj += gold_log_prob
            weight_vector.apply_gradient_update(gradient, 1)
        if not silent:
            print("Objective for epoch: %.2f in time %.2f" % (total_obj, time.time() - epoch_start))
    return CrfNerModel(tag_indexer, feature_indexer, weight_vector)


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)


def compute_gradient(sentence: LabeledSentence, tag_indexer: Indexer, scorer: FeatureBasedSequenceScorer, feature_indexer: Indexer) -> (float, Counter):
    """
    Computes the gradient of the given example (sentence). The bulk of this code will be computing marginals via
    forward-backward: you should first compute these marginals, then accumulate the gradient based on the log
    probabilities.
    :param sentence: The LabeledSentence of the current example
    :param tag_indexer: The Indexer of the tags
    :param scorer: FeatureBasedSequenceScorer is a scoring model that wraps the weight vector and which also contains a
    feat_cache field that will be useful when computing the gradient.
    :param feature_indexer: The Indexer of the features
    :return: A tuple of two items. The first is the log probability of the correct sequence, which corresponds to the
    training objective. This value is only needed for printing, so technically you do not *need* to return it, but it
    will probably be useful to compute for debugging purposes.
    The second value is a Counter containing the gradient -- this is a sparse map from indices (features)
    to weights (gradient values).
    """

    #========================== forward-backward start ==========================
    # find alpha -> forward pass
    log_alpha = np.zeros((len(sentence), len(tag_indexer)))
    for tag_idx in range(0, len(tag_indexer)):
        log_alpha[0][tag_idx] = scorer.score_emission(sentence.tokens[0], tag_idx, 0)
    for word_idx in range(1, len(sentence)):
        for tag_idx in range(0, len(tag_indexer)):
            #log_alpha[word_idx][tag_idx] = -np.inf
            for prev_tag_idx in range(0, len(tag_indexer)):
                log_alpha[word_idx][tag_idx] = np.logaddexp(log_alpha[word_idx][tag_idx], log_alpha[word_idx - 1][prev_tag_idx] + \
                                                                                          scorer.score_emission(sentence.tokens[word_idx], tag_idx, word_idx))
                                                                                          #scorer.score_transition(sentence.tokens[word_idx], prev_tag_idx, tag_idx))     

    #print("===== a =====")
    #print(log_alpha)

    # find beta -> backward pass
    log_beta = np.zeros((len(sentence), len(tag_indexer)))
    for word_idx in range(len(sentence)-2, -1, -1):
        for tag_idx in range(0, len(tag_indexer)):
            #log_beta[word_idx][tag_idx] = -np.inf
            for next_tag_idx in range(0, len(tag_indexer)):
                log_beta[word_idx][tag_idx] = np.logaddexp(log_beta[word_idx][tag_idx], log_beta[word_idx + 1][next_tag_idx] + \
                                                                                        scorer.score_emission(sentence.tokens[word_idx], next_tag_idx, word_idx))
                                                                                        #scorer.score_transition(sentence.tokens[word_idx], tag_idx, next_tag_idx))
    #print("===== b =====")
    #print(log_beta)

    log_marginal_probs = np.zeros((len(sentence), len(tag_indexer)))
    log_marginal_probs = log_alpha + log_beta

    for word_idx in range(0, len(sentence)):
        denom = -np.inf
        for tag_idx in range(0, len(tag_indexer)):
            denom = np.logaddexp(denom, log_marginal_probs[word_idx][tag_idx])
        #print(denom)
        log_marginal_probs[word_idx] -= denom
    total_prob = 0.0
    #print(np.exp(log_marginal_probs))
    """
    for word_idx in range(0, len(sentence)):
        for tag_idx in range(0, len(tag_indexer)):
            denominator[word_idx] += np.logaddexp(log_alpha[word_idx][tag_idx], log_beta[word_idx][tag_idx])
        log_marginal_probs[word_idx] -= denominator[word_idx]
        #for tag_idx in range(0, len(tag_indexer)):
        #    log_marginal_probs[word_idx][tag_idx] = np.exp(log_marginal_probs[word_idx][tag_idx])
    
    """
    #print("===== log_marginal_probs =====")
    #print(log_marginal_probs)
    
    #c = [0 for _ in range(0, len(tag_indexer))]
    #for i in range(0, len(tag_indexer)):
    #    for j in range(0, len(sentence)):
    #        c[i] += log_marginal_probs[j][i]
    #print(c)
    
    
    #for word_idx in range(0, len(sentence)):
    #     for tag_idx in range(0, len(tag_indexer)):
    #        total_prob += log_marginal_probs[word_idx][tag_idx]
    #for word_idx in range(0, len(sentence)):    
    #    for tag_idx in range(0, len(tag_indexer)):
    #        log_marginal_probs[word_idx][tag_idx] = np.exp(log_marginal_probs[word_idx][tag_idx])
    #=========================== forward-backward end ===========================
    log_marginal_probs = np.exp(log_marginal_probs)
    #print(sentence)
    #print(scorer.feat_cache)
    #========================== compute gradient start ==========================
    gradient = Counter()
    pred_counter = Counter()
    for word_idx in range(0, len(sentence)):
        for tag_idx in range(0, len(tag_indexer)):
            for i in scorer.feat_cache[word_idx][tag_idx]:
                pred_counter[i] += log_marginal_probs[word_idx][tag_idx]
        #print(pred_counter)
        gold_tag_idx = tag_indexer.index_of(sentence.get_bio_tags()[word_idx])
        #print(gold_tag_idx)
        for i in scorer.feat_cache[word_idx][gold_tag_idx]:
            gradient[i] += 1.0

    #print(gradient)

    gradient.subtract(pred_counter)
    for g in gradient.keys():
        gradient[g] = gradient[g] * 1.0
    #print(gradient)
    #=========================== compute gradient end ===========================
    """
    #========================== compute gradient start ==========================
    gradient = Counter()
    pred_counter = Counter()
    for word_idx in range(0, len(sentence)):
        for tag_idx in range(0, len(tag_indexer)):
            feat = scorer.feat_cache[word_idx][tag_idx]
            for i in feat:
                pred_counter[i] += np.exp(log_marginal_probs[word_idx][tag_idx])
        #print(pred_counter)
        gold_tag = sentence.get_bio_tags()[word_idx]
        gold_tag_idx = tag_indexer.index_of(gold_tag)
        #print(gold_tag_idx)
        for i in scorer.feat_cache[word_idx][gold_tag_idx]:
            gradient[i] += 1.0

    #print(gradient)

    gradient.subtract(pred_counter)
    #print(gradient)
    #=========================== compute gradient end ===========================
    """
    return total_prob, gradient