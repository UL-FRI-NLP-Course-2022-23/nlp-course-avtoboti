from coref.contextual_model_bert import ContextualControllerBERT
from coref.data import Document
from coref.run_model import to_input

import copy

class Resolver:
    def __init__(self):
        instance = ContextualControllerBERT.from_pretrained("../models/slo_coref")
        instance.eval_mode()
        self.instance = instance

    def coref(self, classla_output, ne_candidates, threshold, window_size=None, window_stride=None):
        if window_size is None:
            window_size = len(classla_output.sentences)
            window_stride = window_size
        if window_stride is None:
            window_stride = window_size

        sentences_data = to_input(classla_output, ne_candidates)

        coreferenced_mentions = set()
        mention_coreferences = {}
        sent_start = 0
        sent_end = min(window_size, len(sentences_data))
        while True:
            print(f"{sent_start}:{sent_end}")
            tokens = {k: copy.deepcopy(v) for sentence in sentences_data[sent_start:sent_end] for k, v in sentence["tokens"].items()}
            sentences = [s for sentence in sentences_data[sent_start:sent_end] for s in sentence["sentences"]]
            mentions = {k: copy.deepcopy(v) for sentence in sentences_data[sent_start:sent_end] for k, v in sentence["mentions"].items()}
            clusters = [c for sentence in sentences_data[sent_start:sent_end] for c in sentence["clusters"]]

            for k in tokens:
                tokens[k].sentence_index -= sent_start
            for k in mentions:
                for token in mentions[k].tokens:
                    token.sentence_index -= sent_start

            iter_input = Document(1, tokens, sentences, mentions, clusters)
            iter_output = self.instance.evaluate_single(iter_input)

            for id2, id1s in iter_output["predictions"].items():
                if id2 is not None:
                    for id1 in id1s:
                        mention_score = iter_output["scores"][id1]
                        if mention_score < threshold:
                            continue
                        if id2 not in mention_coreferences:
                            mention_coreferences[id2] = set()
                            coreferenced_mentions.add(id2)
                        mention_coreferences[id2].add(id1)
                        coreferenced_mentions.add(id1)

            if sent_end == len(sentences_data):
                break
            else:
                sent_start += window_stride
                sent_end = min(sent_end + window_stride, len(sentences_data))
        
        mentions = {k: copy.deepcopy(v) for sentence in sentences_data for k, v in sentence["mentions"].items()}
        return mentions.values(), mention_coreferences