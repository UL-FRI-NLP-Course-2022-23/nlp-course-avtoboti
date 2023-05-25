from common import find_character_sentences


def sentiment_analysis(characters, book_text, afinn, nlp, coref_json=None, find_mode='direct', lang='en'):
    """
    Sentiment analysis using Afinn
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param nlp: stanza pipeline
    :param afinn: Afinn dictionary
    :param coref_json: a JSON object containing co-references and their positions
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

    # Move the mean to 0
    # mean = np.mean(list(sentiments_dir.values()))
    # sentiments_dir = {k: v - mean for k, v in sentiments_dir.items()}

    return sentiments


def afinn_score_slovene(sentence, afinn, nlp):
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
