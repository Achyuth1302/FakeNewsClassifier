import re

def clean_text(text):
    # Lowercase, remove special characters, etc.
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(df):
    df['text'] = df['title'] + " " + df['text']
    df['text'] = df['text'].apply(clean_text)
    return df
