from googletrans import Translator

from common import find_character_sentences


def sentiment_analysis(characters, book_text, sia, nlp, coref_json=None, find_mode='direct', lang="en"):
    """
    Sentiment analysis using VADER
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param sia: VADER sentiment analyzer
    :param nlp: classla or stanza pipeline
    :param coref_json: a JSON object containing co-references and their positions
    :param find_mode: mode of finding sentences ('direct', 'lemma' or 'coref')
    :param lang: language of the book
    :return: dictionary of characters and their sentiments
    """

    # Find sentences containing characters
    characters_sentences = find_character_sentences(characters, book_text, nlp,
                                                    coref_json=coref_json, find_mode=find_mode, lang=lang)

    sentiments = characters.copy()

    translated_characters_sentences = {}
    translator = Translator()

    # If Slovene, translate to English (apparently the sentiment analysis works better with English, see report)
    if lang == "sl":
        # Translate each sentence of each character
        for c, sentences in characters_sentences.items():
            translated_characters_sentences[c] = []
            for sentence in sentences:
                translated_sentence = translator.translate(sentence, src='sl', dest='en').text
                translated_characters_sentences[c].append(translated_sentence)

        characters_sentences = translated_characters_sentences

    for c, sentences in characters_sentences.items():
        sentiments[c] = {
            'frequency': characters[c],
            'sentiment': 0
        }

        for sentence in sentences:
            sentiments[c]['sentiment'] += sia.polarity_scores(sentence)['compound']

        # Average the sentiment
        if len(sentences) > 0:
            sentiments[c]['sentiment'] /= len(sentences)

    # Move the mean to 0
    # mean = np.mean(list(sentiments_dir.values()))
    # sentiments_dir = {k: v - mean for k, v in sentiments_dir.items()}

    return sentiments
