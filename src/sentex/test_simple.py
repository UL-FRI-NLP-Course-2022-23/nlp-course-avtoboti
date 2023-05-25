from nltk.sentiment import SentimentIntensityAnalyzer
import classla

from sentex_nltk import sentiment_analysis

text = "Janez je vesel. Tine je jezen. Janez je videl Petra. Janez in Peter sta najboljša prijatelja. " \
       "Janez in Peter se dobro poznata. Janez ima sovražnika Tineta."

characters = {"Janez": text.count("Janez"), "Peter": text.count("Peter"), "Tine": text.count("Tine")}

# Extract sentiment using NLTK
sia = SentimentIntensityAnalyzer()
nlp = classla.Pipeline('sl', dir='../../models/classla_resources', processors='tokenize,pos,lemma')
sentiments = sentiment_analysis(characters, text, sia, nlp, find_mode='lemma', lang='sl')

# Print results
print(sentiments)

# Results:
# {
#        'Janez': {
#               'frequency': 5,
#               'sentiment': 0.22203999999999996
#        },
#        'Peter': {
#               'frequency': 2,
#               'sentiment': 0.3602
#        },
#        'Tine': {
#               'frequency': 2,
#               'sentiment': -0.5264500000000001
#        }
# }
