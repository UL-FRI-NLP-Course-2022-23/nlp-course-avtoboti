import json
import sys
import os

ENCODINGS = [
    "utf-8",
    "ascii",
    "latin-1"
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
            print("Could not read file '{}' using the {} encoding. Error: {}".format(filename, enc, e), file=sys.stderr)
            print(e, file=sys.stderr)

    if not file_read:
        raise Exception("Could not read file '{}'".format(filename))

    return data


def save_json_file(json_array, filename):
    """
    Saves JSON array to file
    :param json_array: JSON array
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
            print("Could not read file '{}' using the {} encoding. Error: {}".format(filename, enc, e), file=sys.stderr)
            print(e, file=sys.stderr)

    if not file_read:
        raise Exception("Could not read file '{}'".format(filename))

    return data
