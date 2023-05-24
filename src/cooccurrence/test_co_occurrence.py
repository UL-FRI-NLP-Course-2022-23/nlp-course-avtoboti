from src.sentex.sentex_nltk import sentiment_analysis
from co_occurrence_extraction import extract_co_occurrences
from visualization import visualize_connections

from nltk.sentiment import SentimentIntensityAnalyzer
import classla

text = "Janez je videl Petra. Peter je videl Tineta. Janez in Peter sta najboljša prijatelja."
text_en = "Janez is happy. Peter is sad. Tine is angry. Janez and Peter are best friends. " \
          "Janez and Peter love each other. Janez has an enemy Tine."
characters = {"Janez": 4, "Peter": 3, "Tine": 2}

# Extract sentiment using NLTK
sia = SentimentIntensityAnalyzer()
sentiments = sentiment_analysis(characters, text, sia, lang='sl')

# Extract co-occurrences using Classla
nlp = classla.Pipeline('sl', dir='../../models/classla_resources', processors='tokenize,pos,lemma')
co_occurrences = extract_co_occurrences(sentiments, text, nlp, finding_method='direct')

# Print results
print(sentiments)
print(co_occurrences)

# Visualize characters and their connections
visualize_connections(sentiments, co_occurrences)
