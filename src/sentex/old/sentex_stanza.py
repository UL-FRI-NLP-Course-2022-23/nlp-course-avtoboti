import stanza
import classla
import numpy as np

import common


def sentiment_analysis(characters, text, lang="en"):
    """
    :param characters: a json file containing characters
    :param text: book text
    :param lang: language of the book
    :return: a list of character connections and their sentiment
    """

    if lang == "sl":
        classla.download('sl', type='standard_jos')
        nlp = classla.Pipeline('sl', processors='tokenize,sentiment', type='standard_jos')
    else:
        stanza.download('en')
        nlp = stanza.Pipeline('en', processors='tokenize,sentiment')

    doc = nlp(text)

    # Initialize the connections matrix to 0
    character_sentiments = np.zeros(len(characters))

    # Go through all characters
    i = 0
    for c, _ in characters.items():
        num_of_sentences = 0

        for sentence in doc.sentences:
            if c in sentence.text:
                character_sentiments[i] += sentence.sentiment
                num_of_sentences += 1

        # Average the sentiment
        if num_of_sentences > 0:
            character_sentiments[i] /= num_of_sentences

        i += 1

    return character_sentiments

def sentiment_analysis_with_relations(characters, text, nlp):
    """
    :param characters: a json file containing characters
    :param text: book text
    :param nlp: stanza pipeline
    :return: a list of character connections and their sentiment
    """

    doc = nlp(text)

    # Initialize the connections matrix to 0
    character_sentiments = np.zeros((len(characters), len(characters)))

    # Go through all pairs of characters
    i = 0
    for c1, _ in characters.items():
        j = 0
        for c2, _ in characters.items():

            # Skip as the matrix is symmetric
            if j <= i:
                j += 1
                continue

            # Check if characters are part of the sentence; this is a naive approach
            for sentence in doc.sentences:
                current_connections = 0

                if c1 in sentence.text and c2 in sentence.text:
                    character_sentiments[i][j] += sentence.sentiment
                    current_connections += 1

                # Average the sentiment
                if current_connections > 0:
                    character_sentiments[i][j] /= current_connections

            j += 1
        i += 1

    return character_sentiments
