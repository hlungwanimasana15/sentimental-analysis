from transformers import pipeline
import streamlit as st
from rake_nltk import Rake
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Simple sentence splitter
def simple_sentence_tokenizer(text):
    return [sentence.strip() for sentence in text.split('.') if sentence.strip()]

# Function to generate PDF from DataFrame
def generate_pdf(df):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    x = 50
    y = height - 50

    # Table headers
    for col in df.columns:
        c.drawString(x, y, str(col))
        x += 130

    y -= 20

    # Table rows
    for _, row in df.iterrows():
        x = 50
        for value in row:
            text_value = str(value)
            if len(text_value) > 25:
                text_value = text_value[:25] + "..."
            c.drawString(x, y, text_value)
            x += 130
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    buf.seek(0)
    return buf.read()

# Load model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

nlp = load_model()

# UI
st.title("🧠 Sentiment Analysis Dashboard")

st.sidebar.header("Upload or Enter Text")
input_method = st.sidebar.radio("Choose input method:", ["Type Text", "Upload File"])

texts = []
if input_method == "Type Text":
    text_input = st.text_area("Enter multiple reviews (one per line):", height=200)
    if text_input:
        texts = [line.strip() for line in text_input.splitlines() if line.strip()]
else:
    uploaded_file = st.sidebar.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
            texts = [line.strip() for line in content.splitlines() if line.strip()]
        elif file_type == "csv":
            df_uploaded = pd.read_csv(uploaded_file)
            if not df_uploaded.empty:
                st.sidebar.markdown("### CSV Column Selection")
                col_name = st.sidebar.selectbox("Select the column with text to analyze:", df_uploaded.columns)
                texts = df_uploaded[col_name].dropna().astype(str).tolist()

if st.button("Analyze") and texts:
    results = []

    for text in texts:
        analysis = nlp(text)[0]
        label = analysis['label']

        # Map label to sentiment
        if label in ['1 star', '2 stars']:
            sentiment = 'negative'
        elif label == '3 stars':
            sentiment = 'neutral'
        else:
            sentiment = 'positive'

        confidence = round(analysis['score'], 3)

        r = Rake(sentence_tokenizer=simple_sentence_tokenizer)
        r.extract_keywords_from_text(text)
        keywords = r.get_ranked_phrases()

        results.append({
            "Text": text[:100] + ("..." if len(text) > 100 else ""),
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Keywords": ", ".join(keywords[:5])
        })

    df = pd.DataFrame(results)

    st.subheader("📊 Analysis Results")
    st.dataframe(df)

    # Sentiment Distribution
    st.subheader("📈 Sentiment Distribution")
    color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
    sentiment_order = ["positive", "negative", "neutral"]
    sentiment_counts = df["Sentiment"].value_counts().reindex(sentiment_order).fillna(0)

    fig_bar, ax_bar = plt.subplots()
    sentiment_counts.plot(
        kind='bar',
        color=[color_map[s] for s in sentiment_order],
        ax=ax_bar
    )
    ax_bar.set_ylabel("Count")
    ax_bar.set_xlabel("Sentiment")
    ax_bar.set_title("Sentiment Count")
    st.pyplot(fig_bar)

    # Pie Chart
    st.subheader("🧮 Sentiment Percentage")
    fig_pie, ax_pie = plt.subplots()
    non_zero_counts = sentiment_counts[sentiment_counts > 0]
    if not non_zero_counts.empty:
        ax_pie.pie(
            non_zero_counts,
            labels=[s.capitalize() for s in non_zero_counts.index],
            colors=[color_map[s] for s in non_zero_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
    else:
        st.info("No sentiment data to display in pie chart.")

    # Word Cloud
    all_keywords = " ".join(df["Keywords"].tolist())
    if all_keywords.strip():
        st.subheader("☁️ Keyword Word Cloud")
        wc = WordCloud(background_color="white", colormap="viridis").generate(all_keywords)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # Downloads
    st.subheader("📥 Download Results")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")

    pdf = generate_pdf(df)
    st.download_button("Download as PDF", data=pdf, file_name="sentiment_results.pdf", mime="application/pdf")

else:
    st.info("Enter or upload text to begin analysis.")
