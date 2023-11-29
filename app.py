import streamlit as st
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import pandas as pd
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from PIL import Image
import time
import matplotlib.pyplot as plt


stop_words = set(stopwords.words('english'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define a SessionState class to persist state across reruns
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

# Create a session state object
session_state = SessionState(text_input="", analyzed=False)

#cache to load model
@st.cache_data
def get_model():
    try:
        model = DistilBertForSequenceClassification.from_pretrained("JowenPang/SDG-DistilBERT")
        model.to(DEVICE)

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        return tokenizer, model
    except Exception as e:
        st.error(f"An error occurred during model initialization: {str(e)}")
        # You can choose to return default values or raise the exception again
        return None, None

def preprocess_text(text):

    text = text.lower()

    # remove stopwords
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words and len(w)>2]
    text = ' '.join(filtered_sentence)

    # remove punctuation and digit 
    text = re.sub('[^a-zA-Z]', ' ', text)

    # remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = text.strip()

    return text

def get_image(index):
    png_image_path = os.path.join('data/image/', f"goal{index+1}.png")
    jpg_image_path = os.path.join('data/image/', f"goal{index+1}.jpg")

    # Try to open the image as a PNG, and if that fails, try JPEG
    try:
        image = Image.open(png_image_path)
    except FileNotFoundError:
        # If the PNG file is not found, try opening the JPEG file
        try:
            image = Image.open(jpg_image_path)
        except FileNotFoundError:
            # Handle the case where neither PNG nor JPEG file is found
            return None  # Or you can raise an exception or return a default image

    return image

def predict(model, tokenizer, test_sample,device):
    model.eval()

    # Tokenize the test sample
    inputs = tokenizer(test_sample, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Move data to the specified device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class (assuming a classification task)
    # predicted_class = torch.argmax(logits, dim=1).item()
    probabilities = torch.softmax(logits, dim=1)

    top3 = torch.topk(probabilities, 3, dim=1)

    return top3

def get_single_prediction(text_input):
    text = preprocess_text(text_input)
    top3 = predict(model, tokenizer, text, DEVICE)

    top3_index = top3.indices[0].tolist()
    top3_values = top3.values[0].tolist()

    top3_filtered = [[f'Goal {index+1}', str(round(value, 2))] for index, value in zip(top3_index, top3_values) if round(value, 2) > 0]

    first_prediction = []
    second_prediction = []
    third_prediction = []

    if len(top3_filtered) >= 1:
        first_prediction = top3_filtered[0]
    if len(top3_filtered) >= 2:
        second_prediction = top3_filtered[1]
    if len(top3_filtered) >= 3:
        third_prediction = top3_filtered[2]

    return first_prediction, second_prediction, third_prediction


def first_page():
    try:
        text_input = session_state.text_input
        analyzed = session_state.analyzed

        # UI part
        st.markdown('## Single Prediction')


        columns = st.columns([8, 2])

        # Use session_state.text_input as the value for text_area
        text_input = columns[0].text_area('Enter Text to Analyze', value=text_input, key="analyze_input", height=250, disabled=analyzed)

        with columns[1]:
            st.write("")  # Add an empty line
            st.write("")
            analyze_button = st.button('Analyze')
            # refresh_button = st.button("Refresh")

        if analyze_button:

            session_state.text_input = text_input
            session_state.analyzed = True

            if not text_input:
                st.warning("Please enter text to analyze.")
            else:
                
                text = preprocess_text(text_input)
                top3 = predict(model, tokenizer, text, DEVICE)

                top3_index = top3.indices[0].tolist()
                top3_values = top3.values[0].tolist()
                icon_images = [get_image(top3_index[i]) for i in range(len(top3_index)) if round(top3_values[i], 3) > 0]

                st.subheader('Predictions:')
                st.write(f'Number of preprocessed tokens : {len(text)}')

                # Add a text informing the user about the token limit
                if len(text) > 512:
                    st.info("Note: The model will only process the first 512 words due to BERT token limit.")

                sdg_cols = st.columns(5)

                for i in range(len(icon_images)):
                    sdg_cols[i].image(icon_images[i], caption=f"Predicted Label: Goal {top3_index[i]+1}", width=100)
                    sdg_cols[i].write(f"Probability: {round(top3_values[i], 3)}")

                st.success("Analysis completed successfully.")

        # Check if the button is clicked
        # if refresh_button:
        #     session_state.text_input = ""
        #     session_state.analyzed = False
        #     st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def second_page():
    try:
        # UI part
        st.markdown('## Batch Prediction')

        # Your upload Excel and table output page content goes here
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "csv"])

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        if uploaded_file is not None:
            # Check the file extension
            file_extension = uploaded_file.name.split(".")[-1]

            if file_extension == "csv":
                # Read the uploaded CSV file
                data = pd.read_csv(uploaded_file)
                # st.write("Data from CSV file:")

            elif file_extension in ["xlsx", "xls"]:
                # Read the uploaded Excel file
                data = pd.read_excel(uploaded_file)

            else:
                st.write("Unsupported file format. Please upload a CSV or Excel file.")
                return
            
            data = data.astype(str)

            # Display the DataFrame to show column names
            st.subheader("Uploaded File Preview:")
            st.dataframe(data.head())

            # Ask user to select the text column
            text_column = st.selectbox("Select the Text Column", data.columns)

            if text_column not in data.columns:
                st.warning(f"The selected text column '{text_column}' doesn't exist in the file. Please choose a valid column.")
                return

            # Display the selected text column
            st.subheader(f"Selected Text Column: {text_column}")
            
            # Button to confirm the selection
            if st.button("Confirm Selection"):
                data = data[[text_column]]

                start_time = time.time()
                data['First Prediction'], data['Second Prediction'], data['Third Prediction'] = zip(
                    *data[text_column].apply(get_single_prediction))
                end_time = time.time()

                st.subheader('Predictions:')
                time_taken = end_time - start_time
                st.write(f'Time taken for prediction: {time_taken:.2f} seconds')

                # Define the number of rows to display per page
                rows_per_page = 10

                # Calculate the number of pages
                num_pages = len(data) // rows_per_page + 1

                # Create a Streamlit page selection widget
                page_number = st.number_input("Page", 1, num_pages, 1)

                # Calculate the start and end row indices for the current page
                start_idx = (page_number - 1) * rows_per_page
                end_idx = start_idx + rows_per_page

                # Display the DataFrame for the current page
                st.dataframe(data.iloc[start_idx:end_idx])

                csv = convert_df(data)

                button_cols = st.columns(2)
                button_cols[0].download_button(label="Download data as csv", data=csv, file_name="results.csv")
                # refresh_button = button_cols[1].button("Refresh")

                # Check if the button is clicked
                # if refresh_button:
                #     st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def show_model_metrics():
    st.markdown('## Model Metrics & Analysis')

    # Paths to files
    metrics_csv_path = 'exp3/training_metrics_epoch5.csv'
    confusion_matrix_image_path = "exp3/confusion_matrix.png"
    metrics_plot_image_path = "exp3/metrics_plot.png"

    # Check if files exist
    if not os.path.exists(metrics_csv_path):
        st.warning(f"Metrics file '{metrics_csv_path}' not found. Please check the file path.")
        return

    if not os.path.exists(confusion_matrix_image_path):
        st.warning(f"Confusion matrix image '{confusion_matrix_image_path}' not found. Please check the file path.")
        return

    if not os.path.exists(metrics_plot_image_path):
        st.warning(f"Metrics plot image '{metrics_plot_image_path}' not found. Please check the file path.")
        return

    # Read metrics CSV
    try:
        loss_acc_df = pd.read_csv(metrics_csv_path)
    except Exception as e:
        st.error(f"An error occurred while reading the metrics CSV file: {str(e)}")
        return

    st.subheader("Loss and Accuracy")
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Training and Validation Loss
    ax1.plot(loss_acc_df['Epoch'], loss_acc_df['Train Loss'], label='Training Loss')
    ax1.plot(loss_acc_df['Epoch'], loss_acc_df['Valid Loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Training and Validation Accuracy
    ax2.plot(loss_acc_df['Epoch'], loss_acc_df['Train Accuracy'], label='Training Accuracy')
    ax2.plot(loss_acc_df['Epoch'], loss_acc_df['Valid Accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)

    # Display confusion_matrix.png
    st.subheader("Confusion Matrix")
    confusion_matrix_image = Image.open(confusion_matrix_image_path)
    st.image(confusion_matrix_image, caption="Confusion Matrix", use_column_width=True)

    # Display metrics_plot.png
    st.subheader("Metrics Plot")
    metrics_plot_image = Image.open(metrics_plot_image_path)
    st.image(metrics_plot_image, caption="Metrics Plot", use_column_width=True)



st.set_page_config(layout='wide')

# Display introductory text or description
st.title("SDG Classifier")
st.write(
    "Welcome to the SDG Classifier! This app allows you to perform SDG (Sustainable Development Goals) classification on text especially paper abstracts."
    " Choose a navigation tab to get started."
)


# first reload
tokenizer,model = get_model()

selected_tab = st.sidebar.selectbox("Navigation", ["Single Prediction", "Batch Prediction","Model Metrics & Analysis"])

if selected_tab == "Single Prediction":
    first_page()
elif selected_tab == "Batch Prediction":
    second_page()
elif selected_tab == "Model Metrics & Analysis":
    show_model_metrics()

    
