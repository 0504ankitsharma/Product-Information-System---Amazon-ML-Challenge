import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from feature_extraction import extract_features
from model_training import train_model
from qa_function import answer_question

# Load and preprocess data
@st.cache_data
def load_data():
    processed_df = preprocess_data('data-extracted.csv')
    features, vectorizer = extract_features(processed_df)
    labels = processed_df['processed_value']
    model, le = train_model(features, labels)
    return processed_df, features, vectorizer, model, le

def get_product_info(df, group_id):
    product_info = df[df['group_id'] == group_id]
    if product_info.empty:
        return None
    
    info_dict = {}
    for _, row in product_info.iterrows():
        info_dict[row['entity_name']] = row['processed_value']
    
    return info_dict

def format_product_info(info_dict, group_id):
    if info_dict is None:
        return f"No information found for group ID: {group_id}"
    
    formatted_info = f"Product Information (Group ID: {group_id})\n\n"
    for key, value in info_dict.items():
        formatted_info += f"{key.capitalize()}: {value}\n"
    
    return formatted_info

# Streamlit app
def main():
    st.title("Product Information System - Amazon ML Challenge")

    # Load data and train model
    processed_df, features, vectorizer, model, le = load_data()

    # Sidebar for choosing query type
    query_type = st.sidebar.radio("Choose query type:", ["Group ID", "General Question"])

    if query_type == "Group ID":
        group_id = st.text_input("Enter the group ID:")
        if group_id:
            try:
                group_id = int(group_id)
                product_info = get_product_info(processed_df, group_id)
                formatted_info = format_product_info(product_info, group_id)
                st.text(formatted_info)

                # Display image if available
                image_link = processed_df[processed_df['group_id'] == group_id]['image_link'].iloc[0]
                st.image(image_link, caption=f"Product Image (Group ID: {group_id})", use_column_width=True)
            except ValueError:
                st.error("Please enter a valid integer for the group ID.")

    else:  # General Question
        user_question = st.text_input("Ask a question about product attributes:", "What is the height of the product?")
        if st.button("Get Answer"):
            answer = answer_question(user_question, model, vectorizer, le)
            st.success(f"Answer: {answer}")

    # Display all unique attributes
    st.sidebar.subheader("Available Attributes")
    unique_attributes = processed_df['entity_name'].unique()
    st.sidebar.write(", ".join(unique_attributes))

    # Display unique group IDs
    st.sidebar.subheader("Available Group IDs")
    unique_group_ids = processed_df['group_id'].unique()
    st.sidebar.write(", ".join(map(str, unique_group_ids)))

if __name__ == "__main__":
    main()