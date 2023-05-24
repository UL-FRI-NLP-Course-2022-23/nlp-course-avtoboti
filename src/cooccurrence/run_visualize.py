import sys
import os

from utils import *
from visualization import visualize_connections
from common import read_json_file

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Usage: python run_visualize.py <story> <mode>")
    else:
        story_arg = sys.argv[1]
        mode_arg = sys.argv[2]

    if story_arg in ess_characters_dict.keys():
        character_dict = ess_characters_dict
        if mode_arg == 'afinn':
            jsonfile_sent = ess_characters_afinn_json
            jsonfile_cooc = ess_co_occurrences_afinn_json
        elif mode_arg == 'vader':
            jsonfile_sent = ess_characters_vader_json
            jsonfile_cooc = ess_co_occurrences_vader_json
        elif mode_arg == 'bert':
            jsonfile_sent = ess_characters_bert_json
            jsonfile_cooc = ess_co_occurrences_bert_json
    elif story_arg in sn_characters_dict.keys():
        character_dict = sn_characters_dict
        if mode_arg == 'afinn':
            jsonfile_sent = sn_characters_afinn_json
            jsonfile_cooc = sn_co_occurrences_afinn_json
        elif mode_arg == 'vader':
            jsonfile_sent = sn_characters_vader_json
            jsonfile_cooc = sn_co_occurrences_vader_json
        elif mode_arg == 'bert':
            jsonfile_sent = sn_characters_bert_json
            jsonfile_cooc = sn_co_occurrences_bert_json
    elif story_arg in sss_characters_dict.keys():
        character_dict = sss_characters_dict
        if mode_arg == 'afinn':
            jsonfile_sent = sss_characters_afinn_json
            jsonfile_cooc = sss_co_occurrences_afinn_json
        elif mode_arg == 'vader':
            jsonfile_sent = sss_characters_vader_json
            jsonfile_cooc = sss_co_occurrences_vader_json
        elif mode_arg == 'bert':
            jsonfile_sent = sss_characters_bert_json
            jsonfile_cooc = sss_co_occurrences_bert_json

    # Get book characters
    json = read_json_file(jsonfile_sent)
    story_sentiments = json[character_dict.get(story_arg)]

    # Get character co-occurrences
    json = read_json_file(jsonfile_cooc)
    story_co_occurrences = json[character_dict.get(story_arg)]

    # Convert keys from strings to tuples
    story_co_occurrences = {tuple(eval(key)): value for key, value in story_co_occurrences.items()}

    # Visualize connections between characters and their sentiments
    visualize_connections(story_sentiments, story_co_occurrences)
