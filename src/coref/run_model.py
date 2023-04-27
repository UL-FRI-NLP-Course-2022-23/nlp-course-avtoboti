from coref.contextual_model_bert import ContextualControllerBERT
from coref.data import Document, Token, Mention

def classla_output_to_coref_input(classla_output):
    # Transforms CLASSLA's output into a form that can be fed into coref model.
    output_tokens = {}
    output_sentences = []
    output_mentions = {}
    output_clusters = []

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
                output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
                output_clusters.append([current_mention_id])
                mention_tokens = []
                current_mention_id += 1

            # Simplistic mention detection: consider nouns, verbs, adverbs and pronouns as mentions
            if output_token.msd[0] in MENTION_MSD:
                mention_tokens.append(output_token)

            output_tokens[output_token.token_id] = output_token
            output_sentence.append(output_token.token_id)
            token_index_in_document += 1

        # Handle possible leftover mention tokens at end of sentence
        if len(mention_tokens) > 0:
            output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
            output_clusters.append([current_mention_id])
            mention_tokens = []
            current_mention_id += 1

        output_sentences.append(output_sentence)

    return Document(1, output_tokens, output_sentences, output_mentions, output_clusters)

def init_coref():
    instance = ContextualControllerBERT.from_pretrained("../models/slo_coref")
    instance.eval_mode()
    return instance

coref_model = init_coref()

def coref_resolution(classla_output, threshold, return_singletons):
    coref_input = classla_output_to_coref_input(classla_output)
    coref_output = coref_model.evaluate_single(coref_input)
    # 4. prepare response (mentions + coreferences)
    coreferences = []
    coreferenced_mentions = set()
    for id2, id1s in coref_output["predictions"].items():
        if id2 is not None:
            for id1 in id1s:
                mention_score = coref_output["scores"][id1]

                if mention_score < threshold:
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
                "ner_type": classla_output.sentences[sentence_id].tokens[token_id].ner.replace("B-", "").replace("I-", ""),
                "msd": mention.tokens[0].msd,
                "text": mention_raw_text
            }
        )

    return {
        "mentions": mentions,
        "coreferences": sorted(coreferences, key=lambda x: x["id1"])
    }