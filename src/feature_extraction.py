# Implements functions for Tokenization and Encoding with the help of the
# powerful BERT tokenizer to capture the contextual relationships between words

from transformers import BertTokenizer

def get_tokenizer():
    """
    Loads the pretrained BERT tokenizer and returns a transformer object.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def encode_texts(texts, tokenizer, max_length = 128):
    """
    Tokenizes and encodes texts using BERT tokenizer.

    :param texts: List or series of texts.
    :param tokenizer: BertTokenizer instance.
    :param max_length: Maximum length of the encoded sequence.

    :return: The encoded input sequence as a dictionary.
    """
    encoded_inputs = tokenizer(
        list(texts),
        padding='max_length',   # Pads sequences to max_length
        truncation=True,        # Truncates sequences longer than max_length
        max_length=max_length,  # Sets the maximum sequence length
        return_tensors="tf"     # Returns TensorFlow tensors
    )
    return encoded_inputs