import math

import numpy as np

from common import find_character_sentences


def sentiment_analysis(characters, book_text, afinn, nlp, coref_json=None,
                       center_mean=False, find_mode='direct', lang='en'):
    """
    Sentiment analysis using Afinn
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param nlp: stanza pipeline
    :param afinn: Afinn dictionary
    :param coref_json: a JSON object containing co-references and their positions
    :param center_mean: whether to center the mean of the sentiment values to 0
    :param find_mode: mode of finding sentences with characters ('direct', 'lemma' or 'coref')
    :param lang: language of the book
    :return: dictionary of characters and their sentiments
    """

    # Find sentences containing characters
    characters_sentences = find_character_sentences(characters, book_text, nlp,
                                                    coref_json=coref_json, find_mode=find_mode, lang=lang)

    sentiments = characters.copy()

    for c, sentences in characters_sentences.items():
        sentiments[c] = {
            'frequency': characters[c],
            'sentiment': 0
        }

        for sentence in sentences:
            if lang == 'en':
                sentiments[c]['sentiment'] += afinn.score(sentence) / 5
            elif lang == 'sl':
                sentiments[c]['sentiment'] += afinn_score_slovene(sentence, afinn, nlp) / 5

        # Average the sentiment
        if len(sentences) > 0:
            sentiments[c]['sentiment'] /= len(sentences)

        # To account for the fact that some characters are mentioned more often than others and therefore have more
        # probability that their sentiment will be differently classified (and therefore closer to 0 in the end),
        # we multiply the sentiment by the frequency of the character on a logarithmic scale
        # if find_mode == 'coref':  # We only do this for coref mode, because it's the most prone to this problem
        #     sentiments[c]['sentiment'] *= math.log(len(sentences) + 1, 50)

    # Move the mean to 0
    if center_mean:
        mean = np.mean([sentiment['sentiment'] for sentiment in sentiments.values()])
        for c, sentiment in sentiments.items():
            sentiment['sentiment'] -= mean

    return sentiments


def afinn_score_slovene(sentence, afinn, nlp):
    """
    Returns the sentiment score of a sentence using the Afinn dictionary
    :param sentence: sentence to be analyzed
    :param afinn: Slovene Afinn dictionary
    :param nlp: stanza or classla pipeline
    :return: sentiment score
    """

    score = 0
    lemma_count = 0
    doc = nlp(sentence)
    for word in doc.sentences[0].words:
        if word.lemma in afinn:
            word_score = afinn[word.lemma]
            score += word_score
            if word_score != 0:
                lemma_count += 1
    return score / lemma_count if lemma_count > 0 else 0


def create_slovene_afinn_dict():  # For reading the Slovene Afinn dictionary from a file
    afinn_dict = dict()
    # Read book text
    with open('afinn_lex/afinn-slo.txt', 'r', encoding="utf-8") as f:
        # Skip first line
        next(f)
        for line in f:
            line = line.strip()
            line = line.split('\t')
            afinn_dict[line[0]] = int(line[1])
    return afinn_dict
