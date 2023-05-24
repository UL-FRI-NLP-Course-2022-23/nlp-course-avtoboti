import json
import sys
import os

import nltk

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


def remove_infrequent_characters(json, n):
    copy_dict = json.copy()

    for char, freq in copy_dict.items():
        if freq <= n:
            del json[char]

    return json


def remove_false_characters(json, lang="en"):
    copy_dict = json.copy()

    for char, freq in copy_dict.items():
        if lang == "en":
            # Remove characters that are stopwords or are just letters
            if len(char) < 2 or char.lower() in nltk.corpus.stopwords.words('english'):
                del json[char]

            # If it's anything else that we don't want, remove it
            if char in OTHER_CHARACTERS_EN:
                del json[char]

        elif lang == "sl":
            # Remove characters that are stopwords or are just letters
            if len(char) < 2 or char.lower() in nltk.corpus.stopwords.words('slovene'):
                del json[char]

            # If it's anything else that we don't want, remove it
            if char in OTHER_CHARACTERS_SL:
                del json[char]

    # Other characters we will remove manually
    return json
