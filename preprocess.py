import pandas as pd
import re

def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    
    def extract_value_unit(text):
        if pd.isna(text):
            return "Unknown"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(IN|CM|MM|KG|G|LBS|OZ|V|A|HZ|W)',
            r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*(IN|CM|MM)',
            r'(\d+(?:\.\d+)?)\s*(?:X|x)\s*(\d+(?:\.\d+)?)\s*(?:X|x)\s*(\d+(?:\.\d+)?)\s*(IN|CM|MM)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 4:  # For 3D measurements
                    return f"{match.group(1)}x{match.group(2)}x{match.group(3)} {match.group(4)}"
                elif len(match.groups()) == 3:  # For fractions
                    return f"{float(match.group(1))/float(match.group(2))} {match.group(3)}"
                else:
                    return f"{match.group(1)} {match.group(2)}"
        return "Unknown"
    
    df['processed_value'] = df['extracted_text'].apply(extract_value_unit)
    return df

if __name__ == "__main__":
    processed_df = preprocess_data('data-extracted.csv')
    print(processed_df[['entity_name', 'processed_value']])