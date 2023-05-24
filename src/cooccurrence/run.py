import sys
import os

import stanza
import classla

from utils import *

from co_occurrence_extraction import extract_co_occurrences
from common import read_json_file, get_book_text, save_json_file


# Go through all the books in the folder
def extract_all_co_occurrences(stories, mode):
    """
    Extracts co-occurrences for all books in the folder
    :param stories: stories to be analyzed
    :param mode: sentiment analysis mode
    :param infrequent_characters: number of times a character has to appear in the book to be considered
    :return: JSON with sentiment scores for each character for each book
    """

    co_occurrences = []

    # Choose folder
    if stories == 'ess':
        folder = ess_dir
        character_dict = ess_characters_dict
        if 'afinn' in mode:
            jsonfile = ess_characters_afinn_json
        elif 'vader' in mode:
            jsonfile = ess_characters_vader_json
        elif 'bert' in mode:
            jsonfile = ess_characters_bert_json
        else:
            raise Exception("Mode not supported")
        lang = 'en'
    elif stories == 'sn':
        folder = sn_dir
        character_dict = sn_characters_dict
        if 'afinn' in mode:
            jsonfile = sn_characters_afinn_json
        elif 'vader' in mode:
            jsonfile = sn_characters_vader_json
        elif 'bert' in mode:
            jsonfile = sn_characters_bert_json
        else:
            raise Exception("Mode not supported")
        lang = 'sl'
    elif stories == 'sss':
        folder = sss_dir
        character_dict = sss_characters_dict
        if 'afinn' in mode:
            jsonfile = sss_characters_afinn_json
        elif 'vader' in mode:
            jsonfile = sss_characters_vader_json
        elif 'bert' in mode:
            jsonfile = sss_characters_bert_json
        else:
            raise Exception("Mode not supported")
        lang = 'sl'
    else:
        raise Exception("Stories not supported")

    if lang == 'en':
        nlp = stanza.Pipeline('en', processors='tokenize,lemma')
    elif lang == 'sl':
        nlp = classla.Pipeline('sl', dir='../../models/classla_resources', processors='tokenize,pos,lemma')

    # Get book characters
    books_json = read_json_file(jsonfile)

    # Go through all the books in the folder
    for filename in os.listdir(folder):
        print('Extracting co-occurrences from {}'.format(filename))

        # Get characters and their sentiments
        sentiments = books_json[character_dict.get(filename)]

        # No characters are frequent enough, or none are
        # detected by the NER
        if sentiments is None:
            co_occurrences.append(None)
            continue

        # Get book text
        text = get_book_text(os.path.join(folder, filename))

        # Get co-occurrences
        co_occurrences.append(extract_co_occurrences(sentiments, text, nlp))

    return co_occurrences


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Usage: python run.py <stories> <mode>')

    stories_arg = sys.argv[1]
    mode_arg = sys.argv[2]

    stories_co_occurrences = extract_all_co_occurrences(stories_arg, mode_arg)

    # Convert tuple keys to string keys
    for i in range(len(stories_co_occurrences)):
        if stories_co_occurrences[i] is not None:
            stories_co_occurrences[i] = {str(key): value for key, value in stories_co_occurrences[i].items()}
        else:
            stories_co_occurrences[i] = {}

    save_json_file(stories_co_occurrences,
                   os.path.join('..', '..', 'data', stories_arg + '_' + mode_arg + '_co_occurrences.json'))
