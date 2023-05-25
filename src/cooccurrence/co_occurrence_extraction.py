from nltk.tokenize import sent_tokenize, word_tokenize


def extract_co_occurrences(characters, book_text, nlp, coref_json=None, find_mode='direct', lang='en'):
    """
    Extract co-occurrences of characters in the book text. The co-occurrences are extracted in three ways: by directly
    searching for the character name in the sentence, by searching for the character name in the sentence, but
    lemmatizing the sentence first or by choosing the sentences based on existing co-references; this has a
    predisposition that we already have a list of co-references and their positions in the text
    :param characters: characters and their sentiments
    :param book_text: text of the book
    :param nlp: stanza pipeline
    :param coref_json: a JSON object containing co-references and their positions
    :param find_mode: method of finding characters (and co-occurrences) in the text ('direct', 'lemma' or 'coref')
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
    if find_mode == 'direct':
        for c1, _ in characters.items():
            for c2, _ in characters.items():
                # Check if entry is already in the dictionary
                if c1 != c2 and (c2, c1) not in co_occurrences:
                    for sentence in sentences:
                        # If Slovene, remove the last letter so that the name is "normalised" (or at least
                        # has a higher chance of being found)
                        cc1 = c1[:-1] if lang == 'sl' else c1
                        cc2 = c2[:-1] if lang == 'sl' else c2

                        # Tokenize sentence
                        ts = word_tokenize(sentence)

                        # Check if both characters are in the sentence
                        if any(cc1 in word for word in ts) and \
                                any(cc2 in word for word in ts):
                            # Add entry to the dictionary
                            if (c1, c2) not in co_occurrences:
                                co_occurrences[(c1, c2)] = 0

                            co_occurrences[(c1, c2)] += 1  # Increment co-occurrence count

    # 2. By searching for the character name in the sentence, but lemmatizing the sentence first
    elif find_mode == 'lemma':
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
    elif find_mode == 'coref':
        # If English, we have already passed in such text that it already contains co-references replaced with
        # character names
        if lang == 'sl':
            raise Exception('Co-reference resolution not supported for Slovene')
        elif lang != 'en':
            raise Exception('Language not supported')

        # Continue in the same way as in the first case, but with the co-references replaced with character names
        for c1, _ in characters.items():
            for c2, _ in characters.items():
                # Check if entry is already in the dictionary
                if c1 != c2 and (c2, c1) not in co_occurrences:
                    for sentence in sentences:
                        # Tokenize sentence
                        ts = word_tokenize(sentence)

                        # Check if both characters are in the sentence
                        if any(c1 in word for word in ts) and \
                                any(c2 in word for word in ts):
                            # Add entry to the dictionary
                            if (c1, c2) not in co_occurrences:
                                co_occurrences[(c1, c2)] = 0

                            co_occurrences[(c1, c2)] += 1  # Increment co-occurrence count

    return co_occurrences
