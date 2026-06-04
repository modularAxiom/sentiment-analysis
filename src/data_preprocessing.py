# Implements a helper function to remove noise from text data
# while preserving natural language structure for BERT tokenization.

import re

def clean_text(text):
    """
    Cleans text data by:
    - Removing URLS
    - Removing HTML tags
    - Removing extra whitespace
    :param text: Input text to be cleaned.
    :return: Cleaned text.
    """

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove extra whitespace and rejoin tokens
    cleaned_text = ' '.join(text.split())

    return cleaned_text



