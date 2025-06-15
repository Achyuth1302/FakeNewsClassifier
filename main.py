# main.py
from data_preparation import load_data
from text_preprocessing import preprocess_text
from model_training import train_model
from evaluation import evaluate

def run_pipeline():
    df = load_data()
    df = preprocess_text(df)

    X = df['text']
    y = df['label']

    model, X_test, y_test = train_model(X, y)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
