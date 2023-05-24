from nltk.tokenize import sent_tokenize


def extract_co_occurrences(characters, book_text, nlp, finding_method='direct', lang='en'):
    """
    :param characters: characters and their sentiments
    :param book_text: text of the book
    :param nlp: stanza pipeline
    :param finding_method: method of finding characters (and co-occurrences) in the text ('direct', 'lemmatized'
           or 'coref')
    :param lang: language of the book ('en' or 'sl')
    :return: dictionary of bigrams and their sentiments
    """

    # Tokenize text into sentences
    if lang == 'en':
        sentences = sent_tokenize(book_text)
    elif lang == 'sl':
        sentences = sent_tokenize(book_text, 'slovene')
    else:
        raise Exception('Language not supported')

    co_occurrences = {}

    # We can search for characters in three ways:
    # 1. By directly searching for the character name in the sentence
    if finding_method == 'direct':
        for c1, _ in characters.items():
            for c2, _ in characters.items():
                # Check if entry is already in the dictionary
                if c1 != c2 and (c2, c1) not in co_occurrences:
                    for sentence in sentences:
                        # If Slovene, remove the last letter so that the name is "normalised" (or at least
                        # has a higher chance of being found)
                        cc1 = c1[:-1] if lang == 'sl' else c1
                        cc2 = c2[:-1] if lang == 'sl' else c2

                        if cc1 in sentence and cc2 in sentence:
                            if (c1, c2) not in co_occurrences:
                                co_occurrences[(c1, c2)] = 0

                            co_occurrences[(c1, c2)] += 1

    # 2. By searching for the character name in the sentence, but lemmatizing the sentence first
    elif finding_method == 'lemmatized':
        for sentence in sentences: # To avoid lemmatizing the same sentence multiple times it's in top loop here
            # Lemmatize current sentence
            sent = nlp(sentence)
            lemmas = [word.lemma for word in sent.sentences[0].words]

            for c1, _ in characters.items():
                cc1 = nlp(c1).sentences[0].words[0].lemma

                for c2, _ in characters.items():
                    cc2 = nlp(c2).sentences[0].words[0].lemma

                    # Check if entry is already in the dictionary
                    if c1 != c2 and (c2, c1) not in co_occurrences:

                        # Check if both characters are in the sentence
                        if cc1 in lemmas and cc2 in lemmas:
                            if (c1, c2) not in co_occurrences:
                                co_occurrences[(c1, c2)] = 0

                            co_occurrences[(c1, c2)] += 1

    # 3. By choosing the sentences based on existing co-references; this has a
    #    predisposition that we already have a list of co-references (or that we
    #    extract them from the text)
    elif finding_method == 'coref':
        if lang == 'en':
            raise NotImplementedError('This method is not yet implemented')
        elif lang == 'sl':
            raise NotImplementedError('This method is not yet implemented')
        else:
            raise Exception('Language not supported')

    return co_occurrences
