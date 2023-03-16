import sys
import nltk

DOCUMENTS = [
    "../data/english_short_stories/Henry_Red_Chief.txt",
    "../data/english_short_stories/Hills Like White Elephants.txt",
    "../data/english_short_stories/LeiningenVstheAnts.txt",
    "../data/english_short_stories/The Lady or the Tiger Original.txt",
    "../data/english_short_stories/The Most Dangerous Game.txt",
    "../data/english_short_stories/The Tell Tale Heart.txt",
    "../data/english_short_stories/the_gift_of_the_magi_0_Henry.txt"
]

ENCODINGS = [
    "utf-8",
    "ascii",
    "latin-1"
]

def read_file(filename: str) -> str:
    file_read = False
    data = ""
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
    else:
        return data
    
def main():
    for document in DOCUMENTS:
        print(document)
        read_file(document)[:10]

if __name__ == "__main__":
    main()