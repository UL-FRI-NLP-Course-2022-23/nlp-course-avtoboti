import stanza

import sentex_nltk
import sentex_afinn
import sentex_bert

from afinn import Afinn
import classla
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mode = "bert"
sentiment_type = "longer"

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
elif sentiment_type == "very_long":
    text = 'Bil je fant, imenovan Janez, in imel je najboljšega prijatelja Petra. Janez in Peter sta živela v majhni ' \
           'vasici ob vznožju hribov. Bila sta neločljiva in skupaj sta doživela veliko pustolovščin. Nekega sončnega dne ' \
           'sta se Janez in Peter odpravila na raziskovanje gozda. Oblečena v svoje najljubše pustolovske klobuke in s ' \
           'palico v rokah sta prečkala skrivnostno pot. Med hojo sta srečala zajca, ki se je skrival v grmu. Janez je ' \
           'vzkliknil: "Poglej, Peter, zajček! Ali bi rad bil naš prijatelj?" Zajec je zavrtel z ušesi in se približal ' \
           'fantoma. Postal je njihov novi prijatelj in skupaj so nadaljevali pot skozi gozd. Medtem ko so raziskovali, ' \
           'so našli skrivnostno staro hišo. Hiša je izgledala zapuščena in zanimivo je bilo, kaj se skriva v njenem ' \
           'notranjosti. Peter je bil malce prestrašen, vendar je bil Janez pogumen fant. Odločil se je, da bo vstopil ' \
           'v hišo in ugotovil, kaj se skriva znotraj. Janez je odprl vrata in stopil noter, za njim pa so stopili tudi ' \
           'Peter in zajec. V notranjosti hiše so našli staro knjigo, ki je bila polna skrivnosti. Začeli so jo ' \
           'prebirati in odkrili so, da je to knjiga čarovnij. Janez je bil vedno radoveden fant in rekel je: ' \
           '"Poskusimo izpeljati eno čarovnijo!" Fantje so se držali navodil v knjigi in izvedli čarovnijo. Poičakovali ' \
           'so čudež, vendar se ni zgodilo nič. Zdelo se je, da čarovnija ni delovala. Razočarana sta se Janez in Peter ' \
           'odločila, da zapustita hišo. Ko so stopili ven, pa so opazili, da je čarobnost začela delovati. Vse okoli ' \
           'njih so se drevesa začela premikati, cvetje je zacvetelo in vse je bilo polno življenja. Fantje so bili ' \
           'navdušeni nad tem, kar so videli. Zdaj sta vedela, da je prava čarovnija v naravi in v prijateljstvu. ' \
           'Od takrat naprej so Janez, Peter in njihov novi prijatelj zajec preživljali čudovite dni v naravi, ' \
           'odkrivali nove kraje in skupaj ustvarjali nepozabne spomine. In tako se je njihovo prijateljstvo še bolj ' \
           'poglobilo, saj so spoznali, da so najboljše pustolovščine tiste, ki jih doživljajo skupaj.'
    text_en = 'There was a boy named Janez and he had a best friend named Peter. Janez and Peter lived in a small village ' \
              'at the foot of the mountains. They were inseparable and had many adventures together. One sunny day, ' \
              'Janez and Peter set out to explore the forest. Dressed in their favorite adventure hats and armed with ' \
              'sticks, they crossed a mysterious path. While walking, they encountered a rabbit hiding in a bush. Janez ' \
              'exclaimed, "Look, Peter, a bunny! Would you like to be our friend?" The rabbit twitched its ears and ' \
              'approached the boys. It became their new friend, and together they continued their journey through the ' \
              'forest. As they explored, they stumbled upon an old, mysterious house. The house looked abandoned, and ' \
              'they wondered what secrets it held inside. Peter was a little scared, but Janez was a brave boy. He ' \
              'decided to enter the house and find out what lay within. Janez opened the door and stepped inside, with ' \
              'Peter and the rabbit following behind. Inside the house, they discovered an old book filled with ' \
              'secrets. They started reading it and realized it was a book of magic. Curious as ever, Janez said, ' \
              '"Let us try to perform a magic spell!" The boys followed the instructions in the book and attempted ' \
              'the spell. They waited for a miracle, but nothing happened. It seemed that the magic didn not work. ' \
              'Disappointed, Janez and Peter decided to leave the house. As they stepped outside, they noticed that the ' \
              'magic had indeed started working. The trees around them began to move, flowers bloomed, and everything ' \
              'was filled with life. The boys were amazed by what they saw. They now knew that true magic lay in ' \
              'nature and in friendship. From that day on, Janez, Peter, and their new rabbit friend spent wonderful ' \
              'days in nature, discovering new places, and creating unforgettable memories together. And so, their ' \
              'friendship grew even stronger as they realized that the best adventures are the ones they experience ' \
              'together.'
else:
    raise Exception('Story not found.')

characters = {"Janez": text.count("Janez"), "Peter": text.count("Peter")}

nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')
nlp_sl = classla.Pipeline('sl', dir='../../models/classla_resources', processors='tokenize,pos,lemma')

if "afinn" in mode:
    afinn = Afinn(language='en')

    afinn_sl = sentex_afinn.create_slovene_afinn_dict()

    sentiments = sentex_afinn.sentiment_analysis(characters, text, nlp_sl, afinn_sl, lang='sl')
    print("afinn sl: ", sentiments)
    sentiments = sentex_afinn.sentiment_analysis(characters, text_en, nlp, afinn, lang='en')
    print("afinn en: ", sentiments)
elif "vader" in mode:
    sia = SentimentIntensityAnalyzer()

    sentiments = sentex_nltk.sentiment_analysis(characters, text, sia, nlp_sl, lang='sl')
    print("nlkt vader sl: ", sentiments)
    sentiments = sentex_nltk.sentiment_analysis(characters, text_en, sia, nlp, lang='en')
    print("nlkt vader en: ", sentiments)
elif "bert" in mode:
    model_name = 'cjvt/sloberta-sentinews-sentence'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_name_en = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
    model_en = AutoModelForSequenceClassification.from_pretrained(model_name_en)
    device_en = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_en.to(device_en)

    sentiments = sentex_bert.sentiment_analysis(characters, text, model, tokenizer, device, nlp_sl, lang='sl')
    print("bert sl: ", sentiments)
    sentiments = sentex_bert.sentiment_analysis(characters, text_en, model_en, tokenizer_en, device_en, nlp, lang='en')
    print("bert en: ", sentiments)
else:
    raise Exception('Mode not supported.')