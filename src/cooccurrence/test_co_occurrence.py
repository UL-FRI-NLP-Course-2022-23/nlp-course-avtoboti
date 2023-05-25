from co_occurrence_extraction import extract_co_occurrences
from visualization import visualize_connections
import classla

text = "Janez je vesel. Tine je jezen. Janez je videl Petra. Janez in Peter sta najboljša prijatelja. " \
       "Janez ima sovražnika Tineta."

# Obtain results from test_simple.py
sentiments = {
    'Janez': {
        'frequency': 5,
        'sentiment': 0.22203999999999996
    },
    'Peter': {
        'frequency': 2,
        'sentiment': 0.3602
    },
    'Tine': {
        'frequency': 2,
        'sentiment': -0.5264500000000001
    }
}

# Extract co-occurrences using Classla
nlp = classla.Pipeline('sl', dir='../../models/classla_resources', processors='tokenize,pos,lemma')
co_occurrences = extract_co_occurrences(sentiments, text, nlp, find_mode='lemma', lang='sl')

# Print results
print(co_occurrences)

# Visualize characters and their connections
visualize_connections(sentiments, co_occurrences)

# Results:
# {
#     ('Janez', 'Peter'): 2,
#     ('Janez', 'Tine'): 1
# }
