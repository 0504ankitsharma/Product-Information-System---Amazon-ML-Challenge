from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_model(features, labels):
    # Encode the labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    
    return model, le

if __name__ == "__main__":
    from preprocess import preprocess_data
    from feature_extraction import extract_features
    
    processed_df = preprocess_data('data-extracted.csv')
    features, vectorizer = extract_features(processed_df)
    labels = processed_df['processed_value']
    model, le = train_model(features, labels)