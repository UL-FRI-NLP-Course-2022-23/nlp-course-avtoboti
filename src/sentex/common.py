import json
import sys
import os

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

ENCODINGS = [
    "utf-8",
    "ascii",
    "latin-1"
]

OTHER_CHARACTERS_EN = [
    "'s", "O."
]

OTHER_CHARACTERS_SL = [
]


def read_json_file(filename):
    """
    Reads JSON file and returns its contents
    :param filename: path to file
    :return: JSON file contents
    """

    file_read = False
    data = ""

    # Read created JSON file and extract characters
    for enc in ENCODINGS:
        try:
            with open(filename, "r", encoding=enc) as f:
                data = json.load(f)
                file_read = True
                break
        except Exception as e:
            print("Could not read file '{}' using the {} encoding".format(filename, enc), file=sys.stderr)
            print(e, file=sys.stderr)

    if not file_read:
        raise Exception("Could not read file '{}'".format(filename))

    return data


def save_json_file(json_array, filename):
    """
    Saves JSON array to file
    :param json_array: JSON array
    :param filename: file to save to
    """
    if not filename.endswith(".json"):
        filename += ".json"

    # If file exists, ask user if they want to overwrite it
    if os.path.exists(filename):
        overwrite = input("File '{}' already exists. Do you want to overwrite it? (y/n): ".format(filename))
        if overwrite.lower() != 'y':
            return

    with open(filename, 'w') as outfile:
        json.dump(json_array, outfile)


def get_book_text(filename):
    """
    Gets book text given a filename
    :param filename: path to file
    :return: book text
    """

    file_read = False
    data = ""

    # Read book text
    for enc in ENCODINGS:
        try:
            with open(filename, "r", encoding=enc) as f:
                data = f.read()
                file_read = True
                break
        except Exception as e:
            print("Could not read file '{}' using the {} encoding".format(filename, enc), file=sys.stderr)
            print(e, file=sys.stderr)

    if not file_read:
        raise Exception("Could not read file '{}'".format(filename))

    return data


def remove_infrequent_characters(characters_json, n):
    """
    Removes characters that appear less than n times
    :param characters_json: JSON file containing characters
    :param n: number of times a character has to appear in the book to be considered
    :return: JSON file containing characters
    """

    copy_dict = characters_json.copy()

    for char, freq in copy_dict.items():
        if freq < n:
            del characters_json[char]

    return characters_json


def remove_false_characters(characters_json, lang="en"):
    """
    Removes characters that are (presumably) not actual characters (such as stopwords etc.)
    :param characters_json: JSON file containing characters
    :param lang: language of the book
    :return: JSON file containing characters
    """

    copy_dict = characters_json.copy()

    for char, freq in copy_dict.items():
        if lang == "en":
            # Remove characters that are stopwords or are just letters
            if len(char) < 2 or char.lower() in nltk.corpus.stopwords.words('english'):
                del characters_json[char]

            # If it's anything else that we don't want, remove it
            if char in OTHER_CHARACTERS_EN:
                del characters_json[char]

        elif lang == "sl":
            # Remove characters that are stopwords or are just letters
            if len(char) < 2 or char.lower() in nltk.corpus.stopwords.words('slovene'):
                del characters_json[char]

            # If it's anything else that we don't want, remove it
            if char in OTHER_CHARACTERS_SL:
                del characters_json[char]

    # Other characters we will remove manually
    return characters_json


def get_book_corefs(coref_json):
    """
    Gets book coreferences from a JSON file
    :param coref_json: JSON file containing coreferences
    :return: dictionary of coreferences
    """

    corefs = []

    # Recursively go through the JSON file and check if we've reached the lowest level,
    # i.e. the coreferences, in that case add them to the dictionary
    for coref in coref_json:

        # We have not reached the lowest level yet, so recursively call the function
        if type(coref) is list:
            corefs += get_book_corefs(coref)

        # We have reached the lowest level, so add the coreference to the dictionary
        else:
            c = {
                'start': coref['start_char'],
                'end': coref['end_char'],
                'text': coref['text']
            }
            corefs.append(c)

    return corefs


def find_character_sentences(characters, book_text, nlp, coref_json=None, find_mode='direct', lang='en'):
    """
    Finds sentences in which characters appear
    :param characters: dictionary of characters
    :param book_text: the book text
    :param nlp: classla or stanza pipeline
    :param coref_json: a JSON object containing co-references and their positions
    :param find_mode: mode of finding sentences ('direct', 'lemma' or 'coref')
    :return: list of sentences
    """

    characters_sentences = {}

    # 1. Directly search for character names in sentences
    if find_mode == 'direct':
        if lang == 'en':
            # Tokenize text into sentences
            sentences = sent_tokenize(book_text)
        elif lang == 'sl':
            # Tokenize text into sentences
            sentences = sent_tokenize(book_text, 'slovene')
        else:
            raise Exception('Language not supported')

        for c, _ in characters.items():
            characters_sentences[c] = []

            cc = c  # Current character

            # If slovene, remove the last letter so that the name is "normalised"
            if lang == 'sl':
                cc = c[:-1]

            # Find sentences in which the character appears
            for sentence in sentences:
                for word in word_tokenize(sentence):
                    if cc in word:
                        characters_sentences[c].append(sentence)
                        break

    # 2. Search for character lemmas in sentences
    elif find_mode == 'lemma':
        if lang == 'en':
            # Tokenize text into sentences
            sentences = sent_tokenize(book_text)
        elif lang == 'sl':
            # Tokenize text into sentences
            sentences = sent_tokenize(book_text, 'slovene')
        else:
            raise Exception('Language not supported')

        for sentence in sentences:  # To avoid lemmatizing the same sentence multiple times it's in top loop here
            # Lemmatize current sentence
            sent = nlp(sentence)
            lemmas = [word.lemma for word in sent.sentences[0].words]

            for c, _ in characters.items():
                if c not in characters_sentences:
                    characters_sentences[c] = []

                cc = nlp(c).sentences[0].words[0].lemma

                # Check if character lemma is in sentence lemmas
                if cc in lemmas:
                    characters_sentences[c].append(sentence)

    # 3. Search for character co-references in sentences (replacing pronouns with character names) then preforming the
    #    same as in direct mode
    elif find_mode == 'coref':
        # Replace pronouns with character names

        # If Slovene, we already have sentences in which characters appear, so we just need to extract them
        if lang == 'sl':
            # Tokenize text into sentences
            # sentences = sent_tokenize(book_text, 'slovene')

            # Split into sentences using classla
            # sentences = [sentence.text for sentence in nlp(book_text).sentences]
            # Since in our version of classla getting a sentence text is bugged (returns None), we have to use a hack
            # by getting it from _metadata
            sentences = [sentence._metadata.split('# text = ')[1] for sentence in nlp(book_text).sentences]

            for c, _ in characters.items():
                characters_sentences[c] = []

                character_corefs = coref_json[c]

                for coref in character_corefs:
                    try:
                        characters_sentences[c].append(sentences[coref['sentence_index']])
                    except IndexError:
                        pass

        # If English, we have already passed in such text that it already contains co-references replaced with
        # character names
        elif lang == 'en':
            # Tokenize text into sentences
            sentences = sent_tokenize(book_text)

            # Extract sentences in a way similar to direct mode
            for c, _ in characters.items():
                characters_sentences[c] = []

                # Find sentences in which the character appears
                for sentence in sentences:
                    if c in word_tokenize(sentence):
                        characters_sentences[c].append(sentence)
        else:
            raise Exception('Language not supported')



    # If invalid mode, raise exception
    else:
        raise Exception('Invalid find mode')

    return characters_sentences


def replace_text(text, replacements):
    """
    Replaces text in a string with text from a list of replacements
    :param text: string to replace text in
    :param replacements: list of replacements (dictionaries with keys 'start', 'end' and 'text')
    :return: string with replaced text
    """
    offset = 0

    # Replace text with offset
    for replacement in replacements:
        start_index = replacement['start'] + offset
        end_index = replacement['end'] + offset
        replacement_text = replacement['text']

        offset += len(replacement_text) - (end_index - start_index)
        text = text[:start_index] + replacement_text + text[end_index:]

    return text
