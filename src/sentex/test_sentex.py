import sentex_nltk
import sentex_afinn
import sentex_bert

from afinn import Afinn
import classla
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import BertTokenizer, BertForSequenceClassification,\
    AutoTokenizer, AutoModelForSequenceClassification

mode = "vader"
sentiment_type = "longer_negative"

if sentiment_type == "positive":
    text = "Janez je bil vesel. Peter je bil neverjetno vesel."
    text_en = "Janez was happy. Peter was incredibly happy."
elif sentiment_type == "negative":
    text = "Janez je bil žalosten. Peter je bil neverjetno žalosten."
    text_en = "Janez was sad. Peter was incredibly sad."
elif sentiment_type == "neutral":
    text = "Janez je sedel. Peter je stal."
    text_en = "Janez was sitting. Peter was standing."
elif sentiment_type == "mixed":
    text = "Janez je bil vesel. Peter je bil neverjetno žalosten."
    text_en = "Janez was happy. Peter was incredibly sad."
elif sentiment_type == "mixed2":
    text = "Janez je vesel. Peter je žalosten. Janez in Peter sta prijatelja."
    text_en = "Janez is happy. Peter is sad. Janez and Peter are friends."
elif sentiment_type == "longer":
    text = 'Janez je veselo skakal naokoli. Nato je Peter prišel in ga je vprašal, če bi šel na sprehod. ' \
        'Janez je odgovoril, da bi šel z veseljem. Peter je bil vesel, da je Janez sprejel povabilo. ' \
        'Tako sta se odpravila na sprehod. Med sprehodom sta se pogovarjala o vsem mogočem. ' \
        'Janez je bil vesel, da je imel prijatelja, s katerim se lahko pogovarja. '
    text_en = 'Janez was happily jumping around. Then Peter came and asked him if he would go for a walk. ' \
        'Janez replied that he would gladly go. Peter was happy that Janez accepted the invitation. ' \
        'So they went for a walk. During the walk, they talked about everything possible. ' \
        'Janez was happy to have a friend to talk to. '
elif sentiment_type == "longer_negative":
    text = 'Janez je žalostno sedel na klopci. Nato je Peter prišel in ga je vprašal, če bi šel na sprehod. ' \
        'Janez ga je žalostno pogledal in odgovoril, da ne bi šel. Peter je bil žalosten, da Janez ni sprejel povabila. ' \
        'Tako sta se odpravila vsak svojo pot.'
    text_en = 'Janez was sadly sitting on the bench. Then Peter came and asked him if he would go for a walk. ' \
        'Janez looked at him sadly and replied that he would not go. Peter was sad that Janez did not accept the invitation. ' \
        'So they went their separate ways.'

characters = {"Janez": 4, "Peter": 2}

if "afinn_lex" in mode:
    afinn = Afinn(language='en')

    afinn_sl = sentex_afinn.create_slovene_afinn_dict()
    nlp_sl = classla.Pipeline('sl', processors='tokenize,lemma')

    sentiments = sentex_afinn.sentiment_analysis(characters, text, nlp_sl, afinn_sl, lang='sl')
    print("afinn_lex sl: ", sentiments)
    sentiments = sentex_afinn.sentiment_analysis(characters, text_en, None, afinn, lang='en')
    print("afinn_lex en: ", sentiments)
elif "nltk" in mode:
    sia = SentimentIntensityAnalyzer()

    sentiments = sentex_nltk.sentiment_analysis(characters, text, sia, lang='sl')
    print("nlkt vader sl: ", sentiments)
    sentiments = sentex_nltk.sentiment_analysis(characters, text_en, sia, lang='en')
    print("nlkt vader en: ", sentiments)
else:
    model_name = "cjvt/sloberta-sentinews-sentence"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_name_en = 'nlptown/bert-base-multilingual-uncased-sentiment'
    tokenizer_en = BertTokenizer.from_pretrained(model_name_en)
    model_en = BertForSequenceClassification.from_pretrained(model_name_en)
    device_en = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_en.to(device_en)

    sentiments = sentex_bert.sentiment_analysis(characters, text, model, tokenizer, device, lang='sl')
    print("bert sl: ", sentiments)
    sentiments = sentex_bert.sentiment_analysis(characters, text_en, model_en, tokenizer_en, device_en, lang='en')
    print("bert en: ", sentiments)
