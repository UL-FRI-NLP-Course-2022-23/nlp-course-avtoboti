import time

import sentex_afinn
import sentex_nltk
import sentex_bert

from common import *
from utils import *

from afinn import Afinn
import classla
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import BertTokenizer, BertForSequenceClassification,\
    AutoTokenizer, AutoModelForSequenceClassification

MAX_RETRIES = 3


# Go through all the books in the folder
def extract_all_sentiments(stories, mode, infrequent_characters=0):
    """
    Extracts all sentiments from all books in the folder
    :param stories: stories to be analyzed
    :param mode: sentiment analysis mode
    :param infrequent_characters: number of times a character has to appear in the book to be considered
    :return: JSON with sentiment scores for each character for each book
    """

    sentiments = []

    # Choose folder
    if stories == 'ess':
        folder = ess_dir
        jsonfile = ess_characters_json
        character_dict = ess_characters_dict
        lang = 'en'
    elif stories == 'sn':
        folder = sn_dir
        jsonfile = sn_characters_json
        character_dict = sn_characters_dict
        lang = 'sl'
    elif stories == 'sss':
        folder = sss_dir
        jsonfile = sss_characters_json
        character_dict = sss_characters_dict
        lang = 'sl'
    else:
        raise Exception("Stories not supported")

    if "afinn_lex" in mode:
        if lang == 'en':
            afinn = Afinn(language='en')
            nlp = None
        elif lang == 'sl':
            afinn = sentex_afinn.create_slovene_afinn_dict()
            nlp = classla.Pipeline('sl', processors='tokenize,lemma')
        else:
            raise Exception("Language not supported")
    elif "vader" in mode:
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
    elif "bert" in mode:
        # Load pre-trained BERT model and tokenizer
        if lang == 'en':
            model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
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

        # Get book text
        text = get_book_text(os.path.join(folder, filename))

        characters_json = remove_infrequent_characters(characters_json, infrequent_characters)
        characters_json = remove_false_characters(characters_json, lang=lang)

        # If we removed all characters, skip the book
        if characters_json is None or characters_json == {}:
            sentiments.append(None)
            continue

        if "afinn_lex" in mode:
            sentiments.append(sentex_afinn.sentiment_analysis(characters_json, text, nlp, afinn, lang=lang))
        elif "vader" in mode:
            for _ in range(MAX_RETRIES):
                try:
                    sentiments.append(sentex_nltk.sentiment_analysis(characters_json, text, sia, lang=lang))
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
            sentiments.append(sentex_bert.sentiment_analysis(characters_json, text, model, tokenizer, device, lang=lang))

    return sentiments


if __name__ == '__main__':
    stories_arg = sys.argv[1]
    mode_arg = sys.argv[2]

    if len(sys.argv) != 3:
        raise Exception('Usage: python run.py <stories> <mode>')

    stories_sentiments = extract_all_sentiments(stories_arg, mode_arg, 2)
    save_json_file(stories_sentiments,
                   os.path.join('..', '..', 'old/data', stories_arg + '_' + mode_arg + '_sentiments.json'))
