from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(df):
    # Combine entity_name and processed_value
    df['combined_text'] = df['entity_name'] + ' ' + df['processed_value'].fillna('')
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    features = vectorizer.fit_transform(df['combined_text'])
    
    return features, vectorizer

if __name__ == "__main__":
    from preprocess import preprocess_data
    
    processed_df = preprocess_data('data-extracted.csv')
    features, vectorizer = extract_features(processed_df)
    print("Features shape:", features.shape)