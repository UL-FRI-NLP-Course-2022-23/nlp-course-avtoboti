import torch
import numpy as np
from nltk.tokenize import sent_tokenize


def sentiment_analysis(characters, book_text, model, tokenizer, device, lang='en'):
    """
    Sentiment analysis using BERT
    :param characters: dictionary of characters
    :param book_text: text of the book
    :param tokenizer: BERT tokenizer
    :param model: BERT model
    :param device: device to run the model on
    :param lang: language of the book
    :return: dictionary of characters and their sentiments
    """

    if lang == 'en':
        sentences = sent_tokenize(book_text)
        # weights = torch.tensor([-1, -0.5, 0, 0.5, 1]).to(device)
    elif lang == 'sl':
        sentences = sent_tokenize(book_text, 'slovene')
        # weights = torch.tensor([-1, 0, 1]).to(device)
    else:
        raise ValueError('Language not supported')

    sentiments = np.zeros(len(characters))

    i = 0
    for c, _ in characters.items():
        # Current character
        cc = c
        # If slovene, remove the last letter so that the name is "normalised"
        if lang == 'sl':
            cc = c[:-1]

        # Tokenize text and generate input IDs
        input_ids = []
        character_sentences = []
        for sentence in sentences:
            if cc in sentence:
                character_sentences.append(sentence)

                tokens = tokenizer.tokenize(sentence)
                input_id = tokenizer.convert_tokens_to_ids(tokens)
                input_ids.append(torch.tensor(input_id).to(device))

        if len(input_ids) > 0:
            # Pad input IDs
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

            input_ids = torch.tensor(input_ids).to(device)

            # Predict sentiment using BERT model
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

            # Apply softmax to obtain probabilities
            probs = torch.softmax(logits, dim=1)

            for j, _ in enumerate(character_sentences):
                max_prob, sentiment_idx = torch.max(probs[j], dim=0)
                if lang == 'sl':
                    sentiments[i] += sentiment_idx.item() - 1
                    # For printing purposes
                    # sentiment_labels = ['Negative', 'Neutral', 'Positive']
                    # sentiment = sentiment_labels[sentiment_idx.item()]
                    # print(f"Sentence: {sentence}")
                    # print(f"Sentiment: {sentiment} (Confidence: {max_prob.item()})")
                elif lang == 'en':
                    sentiments[i] += (sentiment_idx.item() - 2) / 2
                    # For printing purposes
                    # sentiment_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
                    # sentiment = sentiment_labels[sentiment_idx.item()]
                    # print(f"Sentence: {sentence}")
                    # print(f"Sentiment: {sentiment} (Confidence: {max_prob.item()})")

                # Another, used here, is to compute the average sentiment classifications
                # if lang == 'sl':
                #     sentiments[i] += torch.sum(probs * weights)
                # elif lang == 'en':
                #     sentiments[i] += torch.sum(probs * weights)

        # Average the sentiment
        if len(character_sentences) > 0:
            sentiments[i] /= len(character_sentences)

        i += 1

    # Create a dictionary of characters and their sentiments
    sentiments_dir = dict(zip(characters.keys(), sentiments))

    # Move the mean to 0
    # mean = np.mean(list(sentiments_dir.values()))
    # sentiments_dir = {k: v - mean for k, v in sentiments_dir.items()}

    return sentiments_dir
