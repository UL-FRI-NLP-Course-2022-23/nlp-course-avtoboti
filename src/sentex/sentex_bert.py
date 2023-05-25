import math

import numpy as np
import torch

from common import find_character_sentences


def sentiment_analysis(characters, book_text, model, tokenizer, device, nlp, coref_json=None, find_mode='direct', lang='en'):
    """
    Sentiment analysis using BERT
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param tokenizer: BERT tokenizer
    :param model: BERT model
    :param device: device to run the model on
    :param nlp: classla or stanza pipeline
    :param coref_json: a JSON object containing co-references and their positions
    :param find_mode: mode of finding sentences ('direct', 'lemma' or 'coref')
    :param lang: language of the book
    :return: dictionary of characters and their sentiments
    """

    characters_sentences = find_character_sentences(characters, book_text, nlp,
                                                    coref_json=coref_json, find_mode=find_mode, lang=lang)

    sentiments = characters.copy()

    for c, sentences in characters_sentences.items():
        sentiments[c] = {
            'frequency': characters[c],
            'sentiment': 0
        }

        # Tokenize text and generate input IDs
        input_ids = []
        character_sentences = []

        for sentence in sentences:
            if c in sentence:
                character_sentences.append(sentence)

                tokens = tokenizer.tokenize(sentence)
                input_id = tokenizer.convert_tokens_to_ids(tokens)
                input_ids.append(torch.tensor(input_id).to(device))

        if len(input_ids) > 0:
            # If there's more than 10 sentences, we split them into batches of 10
            if len(input_ids) > 10:
                all_input_ids = [input_ids[i:i + 10] for i in range(0, len(input_ids), 10)]
            else:
                all_input_ids = [input_ids]

            # Go through all batches
            for input_ids in all_input_ids:
                # Pad input IDs
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

                input_ids = torch.tensor(input_ids).to(device)

                # Predict sentiment using BERT model
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits

                # Apply softmax to obtain probabilities
                probs = torch.softmax(logits, dim=1)

                for j, _ in enumerate(input_ids):
                    max_prob, sentiment_idx = torch.max(probs[j], dim=0)
                    if lang == 'sl':
                        sentiments[c]['sentiment'] += sentiment_idx.item() - 1
                        # For printing purposes
                        # sentiment_labels = ['Negative', 'Neutral', 'Positive']
                        # sentiment = sentiment_labels[sentiment_idx.item()]
                        # print(f"Sentence: {sentence}")
                        # print(f"Sentiment: {sentiment} (Confidence: {max_prob.item()})")
                    elif lang == 'en':
                        sentiments[c]['sentiment'] += (sentiment_idx.item() - 2) / 2
                        # For printing purposes
                        # sentiment_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
                        # sentiment = sentiment_labels[sentiment_idx.item()]
                        # print(f"Sentence: {sentence}")
                        # print(f"Sentiment: {sentiment} (Confidence: {max_prob.item()})")

                    # Another, used here, is to compute the average sentiment classifications
                    # if lang == 'sl':
                    #     sentiments[i] += torch.sum(probs * weights)
                    # elif lang == 'en':
                    #     sentiments[i] += torch.sum(probs * weights)

        # Average the sentiment
        if len(sentences) > 0:
            sentiments[c]['sentiment'] /= len(sentences)

        # To account for the fact that some characters are mentioned more often than others and therefore have more
        # probability that their sentiment will be differently classified (and therefore closer to 0 in the end),
        # we multiply the sentiment by the frequency of the character on a logarithmic scale
        # if find_mode == 'coref':  # We only do this for coref mode, because it's the most prone to this problem
        #     sentiments[c]['sentiment'] *= math.log(len(sentences) + 1)

    # Move the mean to 0
    mean = np.mean([sentiment['sentiment'] for sentiment in sentiments.values()])
    for c, sentiment in sentiments.items():
        sentiment['sentiment'] -= mean

    return sentiments
