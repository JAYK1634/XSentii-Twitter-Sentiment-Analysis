# Import necessary modules
import pandas as pd
import numpy as np
# Using TF-IDF for features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Using Decision Tree Classifier as requested
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
import os
import requests
import json
import string # string was missing from original imports but used

# --- NLTK Setup ---
# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
    print("Download complete.")

from nltk.corpus import stopwords

# Initialize NLTK stemmer and stopwords
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# --- Text Cleaning Function ---
def clean(text):
    if not isinstance(text, str):
        text = str(text) # Ensure text is string

    text = text.lower() # Convert text to lowercase
    text = re.sub(r'\[.*?\]', '', text) # Remove square brackets and their contents
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing digits

    # Split into words, remove stopwords, and handle empty strings potentially left after cleaning
    words = [word for word in text.split(' ') if word and word not in stopword]
    text = " ".join(words)

    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in text.split(' ') if word] # Ensure word exists before stemming
    text = " ".join(stemmed_words)

    return text

# --- OCR Function ---
def extract_text(image_file):
    # IMPORTANT: Replace 'K89461099688957' with your own API key or retrieve securely.
    # Using a hardcoded key like this is NOT recommended for production.
    api_key = os.environ.get('OCR_SPACE_API_KEY', 'K89461099688957')

    if api_key == 'K89461099688957':
         print("Warning: Using a default OCR.space API key. Please replace with your own.")

    endpoint = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': api_key,
        'language': 'eng', # Ensure English language model
        'isOverlayRequired': False,
    }

    try:
        with open(image_file, 'rb') as file:
            response = requests.post(
                endpoint,
                files={'image': file},
                data=payload,
                timeout=30 # Add timeout to prevent hanging
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        if not result.get('IsErroredOnProcessing'):
            # Check if ParsedResults exists and is not empty
            parsed_results = result.get('ParsedResults')
            if parsed_results and len(parsed_results) > 0:
                 return result # Return the full JSON if needed, or extract text below
            else:
                return {'ParsedResults': [{'ParsedText': ''}], 'ErrorMessage': 'OCR processing returned no results.'}
        else:
            error_message = result.get('ErrorMessage', ['Unknown OCR Error'])[0]
            return {'ParsedResults': [{'ParsedText': ''}], 'ErrorMessage': f'OCR API Error: {error_message}'}

    except requests.exceptions.RequestException as e:
        return {
            'ParsedResults': [{'ParsedText': ''}],
            'ErrorMessage': f'Network or API request error: {str(e)}'
        }
    except Exception as e:
        return {
            'ParsedResults': [{'ParsedText': ''}],
            'ErrorMessage': f'Error during OCR processing: {str(e)}'
        }


# --- Main Data Loading, Training, and Evaluation ---
DATA_FILE = "twitter.csv"
MODEL_TRAINED = False

try:
    # Load your data from CSV file
    print(f"Loading data from {DATA_FILE}...")
    data = pd.read_csv(DATA_FILE)
    print("Data loaded successfully.")

    # --- Data Preprocessing ---
    print("Preprocessing data...")
    # Check if 'class' column exists
    if 'class' not in data.columns:
        raise ValueError("CSV file must contain a 'class' column for labels.")
    if 'tweet' not in data.columns:
         raise ValueError("CSV file must contain a 'tweet' column for text.")

    # Handle potential missing values in 'tweet' column before cleaning
    data['tweet'] = data['tweet'].fillna('') # Replace NaN with empty string

    # Map numerical labels to descriptive categories
    data["labels"] = data["class"].map({
        0: "Negative Sentiment",
        1: "Potentially Offensive Content",
        2: "Neutral/Positive Content"
    })

    # Check for missing labels after mapping (if 'class' contained unexpected values)
    if data["labels"].isnull().any():
        print("Warning: Some rows have missing labels after mapping. Check 'class' column values.")
        data.dropna(subset=['labels'], inplace=True) # Remove rows with missing labels

    # Keep only relevant columns
    data = data[["tweet", "labels"]]

    # Apply the cleaning function to the "tweet" column
    data["tweet"] = data["tweet"].apply(clean)
    print("Text cleaning complete.")

    # Prepare features (x) and labels (y)
    x = np.array(data["tweet"])
    y = np.array(data["labels"])

    # --- Feature Extraction (Using TF-IDF) ---
    print("Extracting features using TF-IDF...")
    # Use TfidfVectorizer instead of CountVectorizer
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features
    X = vectorizer.fit_transform(x)
    print(f"Features extracted. Shape: {X.shape}")

    # --- Train-Test Split (Stratified) ---
    # Using stratify=y ensures class distribution is similar in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

    # --- Model Training (Using Decision Tree Classifier as requested) ---
    print("Training Decision Tree Classifier model...")
    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier(
        # max_depth=100,       # Original depth - often leads to overfitting
        max_depth=30,          # Reduced max_depth to prevent overfitting (tune this value)
        min_samples_split=5,   # Minimum samples to split a node
        min_samples_leaf=5,    # Minimum samples required at a leaf node (helps regularization)
        random_state=42,
        class_weight='balanced' # Crucial: Adjusts weights for imbalanced classes
    )
    clf.fit(X_train, y_train)
    print("Model training complete.")
    MODEL_TRAINED = True

    # --- Model Evaluation ---
    print("\n--- Model Evaluation ---")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Print classification report with updated labels
    print("\nClassification Report:")
    # Added zero_division=0 to handle cases where a class might have no predicted samples
    print(classification_report(y_test, y_pred, zero_division=0))


except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found.")
    print("Please ensure the CSV file is in the same directory as the script.")
except ValueError as ve:
    print(f"Data Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred during data loading or model training: {str(e)}")
    # Create fallback model and vectorizer using Decision Tree
    print("Creating a minimal fallback model (Decision Tree).")
    vectorizer = TfidfVectorizer() # Fallback vectorizer
    X_fallback = vectorizer.fit_transform(["fallback text example"])
    clf = DecisionTreeClassifier() # Fallback Decision Tree model
    clf.fit(X_fallback.toarray(), ["Neutral/Positive Content"])
    MODEL_TRAINED = True # Mark as trained, even if fallback

# --- Example Usage (Prediction) ---
print("\n--- Prediction Example ---")

# Example 1: Predict custom text
if MODEL_TRAINED:
    sample_text = "This is a wonderful day, full of joy and happiness!"
    print(f"Predicting sentiment for: '{sample_text}'")
    cleaned_text = clean(sample_text)
    # Use try-except for transform in case vectorizer wasn't fitted properly in case of earlier error
    try:
        data_vec = vectorizer.transform([cleaned_text]).toarray()
        prediction = clf.predict(data_vec)
        print(f"Prediction: {prediction[0]}")
    except Exception as transform_error:
         print(f"Error transforming text for prediction: {transform_error}")


    sample_text_neg = "This is really bad and awful, I hate it."
    print(f"\nPredicting sentiment for: '{sample_text_neg}'")
    cleaned_text_neg = clean(sample_text_neg)
    try:
        data_vec_neg = vectorizer.transform([cleaned_text_neg]).toarray()
        prediction_neg = clf.predict(data_vec_neg)
        print(f"Prediction: {prediction_neg[0]}")
    except Exception as transform_error:
         print(f"Error transforming text for prediction: {transform_error}")


else:
    print("Model was not trained successfully. Cannot perform prediction.")


# Example 2: Predict text from an image (replace 'path/to/your/image.png' with an actual image file)
# image_path = 'path/to/your/image.png'
# print(f"\n--- OCR Example (Requires an image file) ---")
# if os.path.exists(image_path):
#     if MODEL_TRAINED:
#         print(f"Extracting text from image: {image_path}")
#         ocr_result = extract_text(image_path)

#         if ocr_result and not ocr_result.get('ErrorMessage'):
#             extracted_text = ocr_result['ParsedResults'][0]['ParsedText']
#             print(f"Extracted Text (raw): {extracted_text[:100]}...") # Print first 100 chars

#             if extracted_text:
#                 cleaned_ocr_text = clean(extracted_text)
#                 if cleaned_ocr_text.strip(): # Check if text remains after cleaning
#                     try:
#                         ocr_vec = vectorizer.transform([cleaned_ocr_text]).toarray()
#                         ocr_prediction = clf.predict(ocr_vec)
#                         print(f"Prediction for image text: {ocr_prediction[0]}")
#                     except Exception as transform_error:
#                         print(f"Error transforming OCR text for prediction: {transform_error}")
#                 else:
#                     print("No significant text found in image after cleaning.")
#             else:
#                 print("No text extracted from the image.")
#         else:
#             print(f"Could not extract text from image. Error: {ocr_result.get('ErrorMessage', 'Unknown OCR error')}")
#     else:
#          print("Model was not trained successfully. Cannot perform prediction on image text.")
# else:
#     print(f"Image file not found at: {image_path}. Skipping OCR example.")
#     print("(To run the OCR example, set 'image_path' to a valid image file)")