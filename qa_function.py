def answer_question(question, model, vectorizer, le):
    # Preprocess the question
    question_features = vectorizer.transform([question])
    
    # Make a prediction
    prediction = model.predict(question_features)[0]
    
    # Decode the prediction
    answer = le.inverse_transform([prediction])[0]
    
    return answer


if __name__ == "__main__":
    from preprocess import preprocess_data
    from feature_extraction import extract_features
    from model_training import train_model
    
    processed_df = preprocess_data('data-extracted.csv')
    features, vectorizer = extract_features(processed_df)
    labels = processed_df['processed_value']
    model, le = train_model(features, labels)
    
    questions = [
        "What is the height of the product?",
        "What is the width of the product?",
        "What is the weight of the item?",
        "What is the voltage of the device?"
    ]

    for question in questions:
        answer = answer_question(question, model, vectorizer, le)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print()