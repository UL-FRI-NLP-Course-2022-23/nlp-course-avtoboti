import time

import sentex_afinn
import sentex_nltk
import sentex_bert

from common import *
from utils import *

from afinn import Afinn
import classla
import stanza
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_RETRIES = 3


# Go through all the books in the folder
def extract_all_sentiments(stories, mode, infrequent_characters=0, find_mode='direct'):
    """
    Extracts all sentiments from all books in the folder
    :param stories: stories to be analyzed
    :param mode: sentiment analysis mode
    :param infrequent_characters: number of times a character has to appear in the book to be considered
    :param find_mode: method of finding characters (and co-occurrences) in the text ('direct', 'lemma' or 'coref')
    :return: JSON with sentiment scores for each character for each book
    """

    sentiments = []
    coref_json = None
    all_coref_json = None  # JSON with co-reference resolution for all books, only needed for coref mode

    # Choose folder
    if stories == 'ess':
        folder = ess_dir
        jsonfile = ess_characters_json
        character_dict = ess_characters_dict
        lang = 'en'
        if find_mode == 'coref':
            all_coref_json = read_json_file(ess_coref_json)
            all_coref_replaced_json = read_json_file(ess_coref_replaced_json)
    elif stories == 'sn':
        folder = sn_dir
        jsonfile = sn_characters_json
        character_dict = sn_characters_dict
        lang = 'sl'
        if find_mode == 'coref':
            all_coref_json = read_json_file(sn_coref_json)
            all_coref_replaced_json = None
    elif stories == 'sss':
        folder = sss_dir
        jsonfile = sss_characters_json
        character_dict = sss_characters_dict
        lang = 'sl'
        if find_mode == 'coref':
            all_coref_json = read_json_file(sss_coref_json)
            all_coref_replaced_json = None
    else:
        raise Exception("Stories not supported")

    if lang == 'sl':
        nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')
    elif lang == 'en':
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')
    else:
        raise Exception('Language not supported')

    if "afinn" in mode:
        if lang == 'en':
            afinn = Afinn(language='en')
        elif lang == 'sl':
            afinn = sentex_afinn.create_slovene_afinn_dict()
        else:
            raise Exception("Language not supported")
    elif "vader" in mode:
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
    elif "bert" in mode:
        # Load pre-trained BERT model and tokenizer
        if lang == 'en':
            model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif lang == 'sl':
            model_name = "cjvt/sloberta-sentinews-sentence"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            raise Exception('Language not supported')

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    else:
        raise Exception("Mode not supported")

    # Get array of characters for all books
    books_json = read_json_file(jsonfile)

    # Go through all the books in the folder
    for filename in os.listdir(folder):
        print('Extracting sentiments from {}...'.format(filename))

        # Get book characters
        characters_json = books_json[character_dict.get(filename)]

        # NER returned no characters
        if characters_json is None or characters_json == {}:
            sentiments.append(None)
            continue

        # Remove infrequent and false characters. Note: if all the characters are infrequent, lower the threshold to
        # the highest frequency of a character in the book
        if all(value < infrequent_characters for value in characters_json.values()):
            min_characters = 0
            for c, freq in characters_json.items():
                if freq > min_characters:
                    min_characters = freq
        else:
            min_characters = infrequent_characters

        # Preform removal of infrequent and false characters
        characters_json = remove_infrequent_characters(characters_json, min_characters)
        characters_json = remove_false_characters(characters_json, lang=lang)

        # Get book text and co-references
        if find_mode == 'coref':
            if lang == 'en':
                text = all_coref_replaced_json[character_dict.get(filename)]
            elif lang == 'sl':
                text = get_book_text(os.path.join(folder, filename))
                coref_json = all_coref_json[character_dict.get(filename)]  # This has indexes of sentences directly
            else:
                raise Exception('Language not supported')
        else:
            # Get book text
            text = get_book_text(os.path.join(folder, filename))

        # If we removed all characters, skip the book
        if characters_json is None or characters_json == {}:
            sentiments.append(None)
            continue

        # If only one character, don't center to mean (since there is only one character)
        center_mean = True
        if len(characters_json) == 1:
            center_mean = False  # We can also decide not to center to mean and just use the character's sentiment

        # Perform sentiment analysis
        if "afinn" in mode:
            sentiments.append(sentex_afinn.sentiment_analysis(characters_json, text, afinn, nlp,
                                                              center_mean=center_mean, coref_json=coref_json,
                                                              find_mode=find_mode, lang=lang))
        elif "vader" in mode:
            for _ in range(MAX_RETRIES):
                try:
                    sentiments.append(sentex_nltk.sentiment_analysis(characters_json, text, sia, nlp,
                                                                     center_mean=center_mean, coref_json=coref_json,
                                                                     find_mode=find_mode, lang=lang))
                    break
                except Exception as e:
                    if _ == MAX_RETRIES - 1:
                        print("Sentiment analysis for {} failed".format(filename))
                        print('Error: {}'.format(e))
                        sentiments.append(None)
                    else:
                        print("Sentiment analysis for {} failed, trying again...".format(filename))
                        # Wait 5 seconds and try again
                        time.sleep(5)
                        continue
        if "bert" in mode:
            sentiments.append(sentex_bert.sentiment_analysis(characters_json, text, model, tokenizer, device, nlp,
                                                             center_mean=center_mean, coref_json=coref_json,
                                                             find_mode=find_mode, lang=lang))

    return sentiments


if __name__ == '__main__':
    stories_arg = sys.argv[1]
    mode_arg = sys.argv[2]

    if len(sys.argv) != 3:
        raise Exception('Usage: python run.py <stories> <mode>')

    stories_sentiments = extract_all_sentiments(stories_arg, mode_arg, infrequent_characters=3, find_mode='coref')

    for i in range(len(stories_sentiments)):
        if stories_sentiments[i] is None:
            stories_sentiments[i] = {}

    save_json_file(stories_sentiments,
                   os.path.join('..', '..', 'data', 'sentiments', stories_arg + '_' + mode_arg + '_sentiments.json'))
