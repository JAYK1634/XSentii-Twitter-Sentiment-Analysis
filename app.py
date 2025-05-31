# Import necessary modules
import streamlit as st
import pyttsx3
import threading
import os
import tempfile
from main import clean, clf, vectorizer, extract_text

# Configure page settings
st.set_page_config(page_title="XSentii - Twitter Sentiment Detection", layout="centered")

# Updated app title and description
st.title("XSentii: Twitter (X) Sentiment Detection")
st.subheader("Enter text or upload an image of a tweet to analyze sentiment.")

# Function to speak text asynchronously in a separate thread
def speak_async_thread(text):
    thread = threading.Thread(target=speak_async, args=(text,))
    thread.daemon = True  # Make thread daemon so it closes with main program
    thread.start()

# Function to speak text asynchronously
def speak_async(text):
    try:
        # Reinitialize the text-to-speech engine each time
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()  # Explicitly stop the engine to release resources
    except Exception as e:
        st.error(f"Error with text-to-speech: {str(e)}")

def find_verdict(user_input):
    try:
        # Clean the user input
        cleaned_input = clean(user_input)

        # Transform the input using the CountVectorizer
        input_data = vectorizer.transform([cleaned_input]).toarray()

        # Predict using the trained model
        verdict = clf.predict(input_data)[0]
        
        # Update the verdict terminology
        if verdict == "Hate Speech":
            verdict = "Negative Sentiment"
        elif verdict == "Offensive Language":
            verdict = "Potentially Offensive Content"
        elif verdict == "No Hate and Offensive":
            verdict = "Neutral/Positive Content"

        # Display the result
        st.success(f"Verdict: {verdict}")

        # Convert the verdict to text for speech
        speech_text = f"The Verdict is {verdict}"

        # Use text-to-speech to say the result asynchronously in a separate thread
        speak_async_thread(speech_text)
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")

# Input for choice - Removed Twitter URL option
input_choice = st.radio("Select Input Type:", ["Text Input", "Upload Image"])

# For text input
if input_choice == 'Text Input':
    user_input = st.text_area("Enter tweet text to analyze:")
    if st.button("Analyze Sentiment"):
        if user_input:
            #  Find result based on the text retrieved
            find_verdict(user_input)
        else:
            # Display a warning if no text is entered
            st.warning("Please enter text before clicking the button.")

# For image input
elif input_choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image of a tweet", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image in the Streamlit app
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Analyze Sentiment'):
            try:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Extract text from the uploaded image using OCR.space API
                extracted_text_response = extract_text(temp_path)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                # Check if the OCR was successful
                if 'ParsedResults' in extracted_text_response and extracted_text_response['ParsedResults']:
                    user_input = extracted_text_response['ParsedResults'][0]['ParsedText']
                    if user_input:
                        st.info(f"Extracted text: {user_input}")
                        find_verdict(user_input)
                    else:
                        st.warning("No text could be extracted from the image.")
                else:
                    st.error("Failed to extract text from the image.")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

# Add footer with additional information
st.markdown("---")
st.markdown("**XSentii** analyzes text to determine sentiment and detect potentially offensive content in tweets.")
