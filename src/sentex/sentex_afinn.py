import numpy as np
from nltk.tokenize import sent_tokenize


def sentiment_analysis(characters, book_text, nlp, afinn, lang='en'):
    """
    Sentiment analysis using Afinn
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param nlp: stanza pipeline
    :param afinn: Afinn dictionary
    :param lang: language of the book
    :return: dictionary of characters and their sentiments
    """

    if lang == 'en':
        # Tokenize text into sentences
        sentences = sent_tokenize(book_text)
    elif lang == 'sl':
        # Tokenize text into sentences
        sentences = sent_tokenize(book_text, 'slovene')
    else:
        raise Exception('Language not supported')

    sentiments = np.zeros(len(characters))

    i = 0
    for c, _ in characters.items():
        # Current character
        cc = c
        # If slovene, remove the last letter so that the name is "normalised"
        if lang == 'sl':
            cc = c[:-1]

        all_occurrences = 0
        for sentence in sentences:
            if cc in sentence:
                all_occurrences += 1
                if lang == 'en':
                    sentiments[i] += afinn.score(sentence) / 5
                elif lang == 'sl':
                    sentiments[i] += afinn_score(sentence, afinn, nlp) / 5

        # Average the sentiment
        if all_occurrences > 0:
            sentiments[i] /= all_occurrences

        i += 1

    # Create a dictionary of characters and their sentiments
    sentiments_dir = dict(zip(characters.keys(), sentiments))

    # Move the mean to 0
    # mean = np.mean(list(sentiments_dir.values()))
    # sentiments_dir = {k: v - mean for k, v in sentiments_dir.items()}

    return sentiments_dir


def afinn_score(sentence, afinn, nlp):
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


def create_slovene_afinn_dict():
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
