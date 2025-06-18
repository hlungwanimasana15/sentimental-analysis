import streamlit as st
from transformers import pipeline
from rake_nltk import Rake
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"


# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Sentence tokenizer workaround for Rake
def simple_sentence_tokenizer(text):
    return [sentence.strip() for sentence in text.split('.') if sentence.strip()]

# Load sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

nlp = load_model()

st.title("🧠 Sentiment Analysis Dashboard")

st.sidebar.header("Upload or Enter Text")
input_method = st.sidebar.radio("Choose input method:", ["Type Text", "Upload .txt File"])

texts = []
# MULTIPLE TEXT INPUT
if input_method == "Type Text":
    text_input = st.text_area(
        "Enter multiple reviews or paragraphs (separate them with a blank line):", height=300
    )
    if text_input:
        # Split by double newline or newlines, keeping paragraph boundaries
        raw_blocks = [block.strip() for block in text_input.split("\n\n") if block.strip()]
        texts = []
        for block in raw_blocks:
            sub_lines = [line.strip() for line in block.splitlines() if line.strip()]
            # Reconstruct full paragraph from lines
            full_text = " ".join(sub_lines)
            texts.append(full_text)

else:
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.splitlines() if line.strip()]

# Generate PDF report
def generate_pdf(dataframe, filename="sentiment_report.pdf"):
    path = f"/mnt/data/{filename}"
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 10)
    text_object = c.beginText(40, height - 40)
    text_object.textLine("Sentiment Analysis Report")
    text_object.textLine("")

    for i, row in dataframe.iterrows():
        text_object.textLine(f"{i + 1}. Sentiment: {row['Sentiment']} | Confidence: {row['Confidence']}")
        text_object.textLine(f"    Text: {row['Text']}")
        text_object.textLine(f"    Keywords: {row['Keywords']}")
        text_object.textLine("")
        if text_object.getY() < 100:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(40, height - 40)

    c.drawText(text_object)
    c.save()
    return path

# MAIN ANALYSIS
if st.button("Analyze") and texts:
    results = []

    for text in texts:
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)

        for sent in sentences:
            if not sent.strip():
                continue

            analysis = nlp(sent)[0]
            label = analysis["label"]
            score = analysis["score"]

            # Simulate NEUTRAL if confidence is low
            if score < 0.6:
                label = "NEUTRAL"

            # Extract keywords
            r = Rake(sentence_tokenizer=simple_sentence_tokenizer)
            r.extract_keywords_from_text(sent)
            keywords = r.get_ranked_phrases()

            results.append({
                "Text": sent[:100] + ("..." if len(sent) > 100 else ""),
                "Sentiment": label,
                "Confidence": round(score, 3),
                "Keywords": ", ".join(keywords[:5])
            })

    df = pd.DataFrame(results)

    st.subheader("📊 Analysis Results")
    st.dataframe(df)

    # Sentiment Distribution
    st.subheader("📈 Sentiment Distribution")
    sentiment_order = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    sentiment_counts = df["Sentiment"].value_counts().reindex(sentiment_order, fill_value=0)
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', color=["green" , "red" , "gray" ], ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    ax.set_title("Sentiment Counts")
    st.pyplot(fig)

    # Word cloud
    all_keywords = " ".join(df["Keywords"].tolist())
    if all_keywords.strip():
        st.subheader("☁️ Keyword Word Cloud")
        wc = WordCloud(background_color="white", colormap="viridis").generate(all_keywords)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # Download buttons
    st.subheader("📥 Download Results")

    # CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download as CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")

    # PDF
    pdf_path = generate_pdf(df)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="sentiment_results.pdf">📄 Download as PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Enter or upload text to begin analysis.")
