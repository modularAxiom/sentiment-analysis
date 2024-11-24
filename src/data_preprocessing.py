# Implements a helper function to remove noise from text data to improve
# model performance and perform text cleaning

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans text data by:
    - Lowercasing
    - Removing URLS
    - Removing HTML tags
    - Removing punctuation
    - Removing numbers
    - Removing extra whitespace
    - Removing stopwords
    - Lemmatization
    :param text: Input text to be cleaned.
    :return: Cleaned text.
    """
    # Lowercase text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = text.strip()

    # Tokenize text
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Rejoin tokens into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text



