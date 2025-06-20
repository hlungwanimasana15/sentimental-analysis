import streamlit as st # Keep this at the top
from transformers import pipeline
from rake_nltk import Rake
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# --- Streamlit UI Configuration (MOVED TO ABSOLUTE TOP) ---
st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard 🧠")


# 1. Load model function definition
@st.cache_resource
def load_model():
    """
    Loads the sentiment analysis model from Hugging Face.
    Uses st.cache_resource to cache the model, preventing re-download
    and re-initialization on every rerun.
    """
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


# 2. Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# 3. Simple sentence splitter (utility function for Rake)
def simple_sentence_tokenizer(text):
    """
    Splits text into sentences based on periods.
    """
    return [sentence.strip() for sentence in text.split('.') if sentence.strip()]

# 4. Function to generate PDF from DataFrame (utility function for downloads)
def generate_pdf(df):
    """
    Generates a PDF document from a pandas DataFrame.
    Includes basic table formatting and page breaks.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Set up basic text properties for PDF
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 30, "Sentiment Analysis Results")
    c.setFont("Helvetica", 10)

    # Calculate column widths more robustly based on available space
    col_widths = {}
    total_width_available = width - 100 # Accounting for left (50) and right (50) margins
    num_columns = len(df.columns)
    if num_columns > 0:
        base_col_width = total_width_available / num_columns
        for col in df.columns:
            col_widths[col] = base_col_width
    else:
        # If no columns, return empty bytes to prevent errors
        c.drawString(50, height - 100, "No data available for PDF.")
        c.save()
        buf.seek(0)
        return buf.read()

    # Starting position for table
    x_start = 50
    y = height - 70
    line_height = 15

    # Draw headers
    current_x = x_start
    for col in df.columns:
        c.drawString(current_x, y, str(col))
        current_x += col_widths[col]
    y -= line_height

    # Draw a line under headers
    c.line(x_start, y + 5, width - 50, y + 5)
    y -= 5 # Adjust for line

    # Draw rows
    for _, row in df.iterrows():
        if y < 50:  # Check if new page is needed (50 units from bottom)
            c.showPage()
            c.setFont("Helvetica", 10) # Reset font after showPage
            y = height - 50 # Reset y for new page
            current_x = x_start
            # Redraw headers on new page
            for col in df.columns:
                c.drawString(current_x, y, str(col))
                current_x += col_widths[col]
            y -= line_height
            c.line(x_start, y + 5, width - 50, y + 5)
            y -= 5 # Adjust for line

        current_x = x_start
        for col_name, value in row.items(): # Iterate with column names to get correct width
            text_value = str(value)
            # Adjust text to fit cell, truncate if too long
            # Rough estimate: 6 pixels per character, adjust as needed
            max_char_per_cell = int(col_widths[col_name] / 6)
            if len(text_value) > max_char_per_cell:
                text_value = text_value[:max_char_per_cell - 3] + "..."
            c.drawString(current_x, y, text_value)
            current_x += col_widths[col_name]
        y -= line_height

    c.save()
    buf.seek(0)
    return buf.read()


# 5. Initialize session state variables (after page config, before other widgets)
if 'text_input_area_value' not in st.session_state:
    st.session_state['text_input_area_value'] = ""
if 'uploaded_file_value' not in st.session_state:
    st.session_state['uploaded_file_value'] = None
if 'analysis_performed' not in st.session_state:
    st.session_state['analysis_performed'] = False
if 'analysis_df' not in st.session_state:
    st.session_state['analysis_df'] = pd.DataFrame()


# 6. Call load_model AFTER its definition and session state initialization
nlp = load_model()

# --- Rest of your Streamlit UI components ---
st.title("🧠 Sentiment Analysis Dashboard")
st.markdown("---") # Visual separator

# --- Sidebar for Configuration ---
st.sidebar.header("Configuration")

# Input Method Radio Buttons
input_method = st.sidebar.radio("Choose input method:", ["Type Text", "Upload File"], key="input_method_radio")

# List to hold texts for analysis
texts = []

# Text Input Area (Conditionally displayed based on radio button)
if input_method == "Type Text":
    text_input = st.text_area(
        "Enter multiple reviews (one per line):",
        height=200,
        key="text_input_area",
        value=st.session_state['text_input_area_value'],
        placeholder="E.g.,\nThis product is amazing!\nThe service was terrible.\nNeutral experience.",
        help="Each line will be treated as a separate review/entry for analysis."
    )
    if text_input != st.session_state['text_input_area_value']:
        st.session_state['text_input_area_value'] = text_input
    
    texts = [line.strip() for line in text_input.splitlines() if line.strip()]

else: # Upload File (Conditionally displayed)
    uploaded_file = st.sidebar.file_uploader(
        "Upload a .txt or .csv file",
        type=["txt", "csv"],
        key="file_uploader",
        help="For CSV, select the column containing text data below."
    )
    st.session_state['uploaded_file_value'] = uploaded_file

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
            texts = [line.strip() for line in content.splitlines() if line.strip()]
        elif file_type == "csv":
            df_uploaded = pd.read_csv(uploaded_file)
            if not df_uploaded.empty:
                st.sidebar.markdown("### CSV Column Selection")
                col_name = st.sidebar.selectbox(
                    "Select the column with text to analyze:",
                    df_uploaded.columns,
                    key="csv_col_selector",
                    help="Choose the column that contains the text you want to analyze sentiment for."
                )
                texts = df_uploaded[col_name].dropna().astype(str).tolist()
            else:
                st.sidebar.warning("The uploaded CSV file is empty.")


st.sidebar.markdown("---")

# Keyword Exclusion Feature (Stopwords)
st.sidebar.subheader("⚙️ Keyword Exclusion Settings")
custom_stopwords_input = st.sidebar.text_input(
    "Enter words to exclude from keywords (comma-separated):",
    value="customer, service, product, really, very, good, bad",
    key="custom_stopwords_input",
    help="These words will be ignored when extracting keywords. Useful for common, uninformative words."
)
additional_stopwords = [word.strip().lower() for word in custom_stopwords_input.split(',') if word.strip()]


# Clear Button in Sidebar
st.sidebar.markdown("---")
def clear_inputs_callback():
    st.session_state['text_input_area_value'] = ""
    st.session_state['file_uploader'] = None
    
    if 'csv_col_selector' in st.session_state:
        del st.session_state['csv_col_selector']
    
    st.session_state['analysis_performed'] = False
    if 'analysis_df' in st.session_state:
        del st.session_state['analysis_df']

st.sidebar.button("🗑️ Clear Inputs", key="clear_button", on_click=clear_inputs_callback, help="Clear the text area or uploaded file and reset results.")

# --- Main Area: Analysis Trigger and Results ---

if st.button("🚀 Analyze Sentiment", key="analyze_button", type="primary", use_container_width=True):
    if not texts:
        st.warning("Please provide some text to analyze via typing or uploading a file.")
        st.session_state['analysis_performed'] = False
    else:
        with st.spinner("Analyzing sentiment and extracting keywords... This may take a moment."):
            results = []

            combined_rake_stopwords = set(nltk.corpus.stopwords.words('english'))
            combined_rake_stopwords.update(additional_stopwords)
            
            r = Rake(sentence_tokenizer=simple_sentence_tokenizer, 
                     stopwords=list(combined_rake_stopwords))

            for i, text in enumerate(texts):
                try:
                    analysis = nlp(text)[0]
                    label = analysis['label']

                    if label in ['1 star', '2 stars']:
                        sentiment = 'Negative'
                    elif label == '3 stars':
                        sentiment = 'Neutral'
                    else:
                        sentiment = 'Positive'

                    confidence = round(analysis['score'], 3)

                    r.extract_keywords_from_text(text)
                    keywords = r.get_ranked_phrases()
                    
                    final_keywords = [
                        kw for kw in keywords 
                        if not any(stop_word == kw.lower() for stop_word in additional_stopwords)
                    ]

                    results.append({
                        "Original Text": text,
                        "Preview": text[:100] + ("..." if len(text) > 100 else ""),
                        "Sentiment": sentiment,
                        "Confidence": confidence,
                        "Keywords": ", ".join(final_keywords[:5]) if final_keywords else "N/A"
                    })
                except Exception as e:
                    st.error(f"Error analyzing text '{text[:50]}...': {e}")
                    results.append({
                        "Original Text": text,
                        "Preview": text[:100] + ("..." if len(text) > 100 else ""),
                        "Sentiment": "Error",
                        "Confidence": 0.0,
                        "Keywords": "Error during analysis"
                    })

        df = pd.DataFrame(results)
        st.session_state['analysis_df'] = df
        st.session_state['analysis_performed'] = True

if not st.session_state.get('analysis_performed', False):
    st.info("Enter or upload text using the sidebar, then click 'Analyze Sentiment' to view results.")

if st.session_state.get('analysis_performed', False):
    df = st.session_state['analysis_df']

    st.markdown("---")
    st.header("📊 Analysis Results")

    st.subheader("Detailed Analysis Table")
    st.dataframe(df, use_container_width=True)

    with st.expander("View Full Text for Each Entry", expanded=False):
        for idx, row in df.iterrows():
            st.write(f"**Entry {idx+1} - Sentiment: {row['Sentiment']}**")
            st.write(row["Original Text"])
            st.markdown("---")

    st.markdown("---")
    st.header("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        color_map = {"Positive": "green", "Negative": "red", "Neutral": "gray", "Error": "purple"}
        sentiment_order = ["Positive", "Negative", "Neutral", "Error"]
        sentiment_counts = df["Sentiment"].value_counts().reindex(sentiment_order).fillna(0)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        sentiment_counts.plot(
            kind='bar',
            color=[color_map[s] for s in sentiment_order if s in sentiment_counts.index],
            ax=ax_bar
        )
        ax_bar.set_ylabel("Count")
        ax_bar.set_xlabel("Sentiment")
        ax_bar.set_title("Sentiment Count")
        plt.xticks(rotation=0)
        st.pyplot(fig_bar)

    with col2:
        st.subheader("Sentiment Percentage")
        fig_pie, ax_pie = plt.subplots(figsize=(8, 5))
        non_zero_counts = sentiment_counts[sentiment_counts > 0]
        if not non_zero_counts.empty:
            ax_pie.pie(
                non_zero_counts,
                labels=[s for s in non_zero_counts.index],
                colors=[color_map[s] for s in non_zero_counts.index],
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12},
                pctdistance=0.85
            )
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
        else:
            st.info("No sentiment data to display in pie chart.")

    st.markdown("---")
    st.subheader("☁️ Keyword Word Cloud")
    all_keywords = " ".join(df["Keywords"].dropna().tolist())
    
    final_wordcloud_stopwords = set(STOPWORDS)
    final_wordcloud_stopwords.update(nltk.corpus.stopwords.words('english'))
    final_wordcloud_stopwords.update(additional_stopwords)

    if all_keywords.strip() and all_keywords != "N/A":
        wc = WordCloud(
            background_color="white",
            colormap="viridis",
            width=800,
            height=400,
            stopwords=final_wordcloud_stopwords,
            collocations=True
        ).generate(all_keywords)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("No keywords to display in the word cloud or keywords were entirely filtered out.")

    st.markdown("---")
    st.header("📥 Download Results")
    st.markdown("Download your analysis results in different formats.")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv",
            help="Download the full table as a CSV file."
        )
    with col_dl2:
        pdf = generate_pdf(df)
        st.download_button(
            label="Download as PDF",
            data=pdf,
            file_name="sentiment_results.pdf",
            mime="application/pdf",
            help="Download the results table as a PDF document."
        )
        
