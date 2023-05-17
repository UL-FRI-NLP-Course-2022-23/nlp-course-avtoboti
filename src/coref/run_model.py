from coref.contextual_model_bert import ContextualControllerBERT
from coref.data import Document, Token, Mention
import copy


def to_input(classla_output):
    # Transforms CLASSLA's output into a form that can be fed into coref model.
    sentences = [{"tokens": {}, "sentences": [], "mentions": {}, "clusters": []} for _ in range(len(classla_output.sentences))]

    str_document = classla_output.text
    start_char = 0
    MENTION_MSD = {"N", "V", "R", "P"}  # noun, verb, adverb, pronoun

    current_mention_id = 1
    token_index_in_document = 0
    for sentence_index, input_sentence in enumerate(classla_output.sentences):
        output_sentence = []
        mention_tokens = []
        for token_index_in_sentence, input_token in enumerate(input_sentence.tokens):
            input_word = input_token.words[0]
            output_token = Token(str(sentence_index) + "-" + str(token_index_in_sentence),
                                 input_word.text,
                                 input_word.lemma,
                                 input_word.xpos,
                                 sentence_index,
                                 token_index_in_sentence,
                                 token_index_in_document)

            # FIXME: This is a possibly inefficient way of finding start_char of a word. Stanza has this functionality
            #  implemented, Classla unfortunately does not, so we resort to a hack
            new_start_char = str_document.find(input_word.text, start_char)
            output_token.start_char = new_start_char
            if new_start_char != -1:
                start_char = new_start_char

            if len(mention_tokens) > 0 and mention_tokens[0].msd[0] != output_token.msd[0]:
                sentences[sentence_index]["mentions"][current_mention_id] = Mention(current_mention_id, mention_tokens)
                sentences[sentence_index]["clusters"].append([current_mention_id])
                mention_tokens = []
                current_mention_id += 1

            # Simplistic mention detection: consider nouns, verbs, adverbs and pronouns as mentions
            if output_token.msd[0] in MENTION_MSD:
                mention_tokens.append(output_token)

            sentences[sentence_index]["tokens"][output_token.token_id] = output_token
            output_sentence.append(output_token.token_id)
            token_index_in_document += 1

        # Handle possible leftover mention tokens at end of sentence
        if len(mention_tokens) > 0:
            sentences[sentence_index]["mentions"][current_mention_id] = Mention(current_mention_id, mention_tokens)
            sentences[sentence_index]["clusters"].append([current_mention_id])
            mention_tokens = []
            current_mention_id += 1

        sentences[sentence_index]["sentences"].append(output_sentence)

    return sentences

def get_response(coref_input, coref_output, classla_data, threshold, return_singletons):
    coreferences = []
    coreferenced_mentions = set()
    for id2, id1s in coref_output["predictions"].items():
        if id2 is not None:
            for id1 in id1s:
                mention_score = coref_output["scores"][id1]

                if mention_score < threshold:
                    # print(f"Denied {id1}, {id2}: {mention_score}")
                    continue

                coreferenced_mentions.add(id1)
                coreferenced_mentions.add(id2)

                coreferences.append({
                    "id1": int(id1),
                    "id2": int(id2),
                    "score": mention_score
                })

    mentions = []
    for mention in coref_input.mentions.values():
        [sentence_id, token_id] = [int(idx) for idx in mention.tokens[0].token_id.split("-")]
        mention_score = coref_output["scores"][mention.mention_id]

        if return_singletons is False and mention.mention_id not in coreferenced_mentions:
            continue

        # while this is technically already filtered with coreferenced_mentions, singleton mentions aren't, but they
        # have some score too that can be thresholded.
        #if req_body.threshold is not None and mention_score < req_body.threshold:            
        #    continue

        mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
        mentions.append(
            {
                "id": mention.mention_id,
                "start_idx": mention.tokens[0].start_char,
                "length": len(mention_raw_text),
                "ner_type": classla_data.sentences[sentence_id].tokens[token_id].ner.replace("B-", "").replace("I-", ""),
                "msd": mention.tokens[0].msd,
                "text": mention_raw_text
            }
        )

    return {
        "mentions": mentions,
        "coreferences": sorted(coreferences, key=lambda x: x["id1"])
    }

def init_coref():
    instance = ContextualControllerBERT.from_pretrained("../models/slo_coref")
    instance.eval_mode()
    return instance

coref_model = init_coref()

def get_key(coreference_map, k):
    if k not in coreference_map:
        return k
    else:
        return get_key(coreference_map, coreference_map[k])

def get_mentions(ms):
    mentions = []
    for mention in ms.values():
        mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
        mentions.append(
            {
                "id": mention.mention_id,
                "start_idx": mention.tokens[0].start_char,
                "length": len(mention_raw_text),
                "msd": mention.tokens[0].msd,
                "text": mention_raw_text
            }
        )
    return mentions

def get_slocoref(classla_output):
    sentences_data = to_input(classla_output)
    all_tokens = {k: v for sentence in sentences_data for k, v in sentence["tokens"].items()}
    all_sentences = [s for sentence in sentences_data for s in sentence["sentences"]]
    all_mentions = {k: v for sentence in sentences_data for k, v in sentence["mentions"].items()}
    all_clusters = [c for sentence in sentences_data for c in sentence["clusters"]]
    coref_input = Document(1, all_tokens, all_sentences, all_mentions, all_clusters)
    return coref_model.evaluate_single(coref_input), get_mentions(all_mentions)

def coref_resolution(classla_output, threshold, return_singletons, window_size=None, window_stride=None):
    sentences_data = to_input(classla_output)
    all_tokens = {k: v for sentence in sentences_data for k, v in sentence["tokens"].items()}
    all_sentences = [s for sentence in sentences_data for s in sentence["sentences"]]
    all_mentions = {k: v for sentence in sentences_data for k, v in sentence["mentions"].items()}
    all_clusters = [c for sentence in sentences_data for c in sentence["clusters"]]

    print(all_sentences)
    print(all_mentions)

    coreferences_map = {}
    ret_mentions = []
    for mention in all_mentions.values():
        [sentence_id, token_id] = [int(idx) for idx in mention.tokens[0].token_id.split("-")]

        mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
        ret_mentions.append(
            {
                "id": mention.mention_id,
                "start_idx": mention.tokens[0].start_char,
                "length": len(mention_raw_text),
                "ner_type": classla_output.sentences[sentence_id].tokens[token_id].ner.replace("B-", "").replace("I-", ""),
                "msd": mention.tokens[0].msd,
                "text": mention_raw_text
            }
        )
    print("----------")
    if window_size is None:
        coref_input = Document(1, all_tokens, all_sentences, all_mentions, all_clusters)
        coref_output = coref_model.evaluate_single(coref_input)
        print(coref_output)
        ret_response = get_response(coref_input, coref_output, classla_output, threshold, return_singletons)
        ret_coreferences = ret_response["coreferences"]
        for coref in ret_coreferences:
            id1 = coref["id1"]
            id2 = coref["id2"]
            min_id = min(id1, id2)
            max_id = max(id1, id2)
            min_id_key = get_key(coreferences_map, min_id)
            max_id_key = get_key(coreferences_map, max_id)
            coreferences_map[max_id] = min(min_id_key, max_id_key)
            if max_id_key < min_id_key and max_id_key != min_id:
                coreferences_map[min_id] = max_id_key
    else:
        if window_stride is None:
            window_stride = window_size
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
            iter_output = coref_model.evaluate_single(iter_input)
            ret_data = get_response(iter_input, iter_output, classla_output, threshold, return_singletons)
            iter_ret_coreferences = ret_data["coreferences"]
            print("inserting_coreferences")
            for coref in iter_ret_coreferences:
                id1 = coref["id1"]
                id2 = coref["id2"]
                min_id = min(id1, id2)
                max_id = max(id1, id2)
                min_id_key = get_key(coreferences_map, min_id)
                max_id_key = get_key(coreferences_map, max_id)
                coreferences_map[max_id] = min(min_id_key, max_id_key)
                if max_id_key < min_id_key and max_id_key != min_id:
                    coreferences_map[min_id] = max_id_key

            if sent_end == len(sentences_data):
                break
            else:
                sent_start += window_stride
                sent_end = min(sent_end + window_stride, len(sentences_data))


    print("Resolving coreference sets")
    # print(coreferences_map)
    coreferences = {k: [k] for k in set(coreferences_map.values())}
    for k, v in coreferences_map.items():
        coreferences[v].append(k)

    return ret_mentions, list(coreferences.values())
    # coref_output = coref_model.evaluate_single(coref_input)
    # # 4. prepare response (mentions + coreferences)
    # coreferences = []
    # coreferenced_mentions = set()
    # for id2, id1s in coref_output["predictions"].items():
    #     if id2 is not None:
    #         for id1 in id1s:
    #             mention_score = coref_output["scores"][id1]

    #             if mention_score < threshold:
    #                 continue

    #             coreferenced_mentions.add(id1)
    #             coreferenced_mentions.add(id2)

    #             coreferences.append({
    #                 "id1": int(id1),
    #                 "id2": int(id2),
    #                 "score": mention_score
    #             })

    # mentions = []
    # for mention in coref_input.mentions.values():
    #     [sentence_id, token_id] = [int(idx) for idx in mention.tokens[0].token_id.split("-")]
    #     mention_score = coref_output["scores"][mention.mention_id]

    #     if return_singletons is False and mention.mention_id not in coreferenced_mentions:
    #         continue

    #     # while this is technically already filtered with coreferenced_mentions, singleton mentions aren't, but they
    #     # have some score too that can be thresholded.
    #     #if req_body.threshold is not None and mention_score < req_body.threshold:            
    #     #    continue

    #     mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
    #     mentions.append(
    #         {
    #             "id": mention.mention_id,
    #             "start_idx": mention.tokens[0].start_char,
    #             "length": len(mention_raw_text),
    #             "ner_type": classla_output.sentences[sentence_id].tokens[token_id].ner.replace("B-", "").replace("I-", ""),
    #             "msd": mention.tokens[0].msd,
    #             "text": mention_raw_text
    #         }
    #     )

    # return {
    #     "mentions": mentions,
    #     "coreferences": sorted(coreferences, key=lambda x: x["id1"])
    # }