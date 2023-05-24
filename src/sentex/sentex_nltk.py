from nltk.tokenize import sent_tokenize
from googletrans import Translator
import numpy as np


def sentiment_analysis(characters, book_text, sia, lang="en"):
    """
    Sentiment analysis using VADER
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param sia: VADER sentiment analyzer
    :param lang: language of the book
    :return: dictionary of characters and their sentiments
    """

    translator = Translator()

    # If Slovene, translate to English (apparently the sentiment analysis works better with English, see report)
    if lang == "sl":
        try:
            book_text = translator.translate(book_text, src='sl', dest='en').text

            # Go throught text and place a space after each dot
            book_text = book_text.replace(".", ". ")

            # Tokenize text into sentences
            sentences = sent_tokenize(book_text)
        except Exception as e:
            # The text may have been too long, split it into sentences and translate each sentence
            sentences = sent_tokenize(book_text, language='slovene')
            translated_sentences = []

            for sentence in sentences:
                translated_sentences.append(translator.translate(sentence, src='sl', dest='en').text)

            sentences = translated_sentences

        # Because it translates names as well, we need to create a map of names and their translations.
        # We assume that the names are will be translated the same way every time
        names = []
        for c, _ in characters.items():
            translated_name = translator.translate(c, src='sl', dest='en').text
            names.append(translated_name)
    elif lang == "en":
        names = characters.keys()

        # Tokenize text into sentences
        sentences = sent_tokenize(book_text)
    else:
        raise ValueError("Language not supported")

    sentiments = np.zeros(len(characters))

    i = 0
    for c in names:
        # Current character
        cc = c
        # If slovene, remove the last letter so that the name is "normalised"
        if lang == 'sl':
            cc = c[:-1]

        all_occurrences = 0
        for sentence in sentences:
            if cc in sentence:
                all_occurrences += 1
                sentiments[i] += sia.polarity_scores(sentence)['compound']

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
