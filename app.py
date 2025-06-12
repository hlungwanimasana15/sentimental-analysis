import streamlit as st
from transformers import pipeline
from rake_nltk import Rake
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk


nltk.download('stopwords')  # <- Add this line near the top
nltk.download('punkt')

# Load sentiment pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

nlp = load_model()

# Title
st.title("🧠 Sentiment Analysis Dashboard")

# Sidebar
st.sidebar.header("Upload or Enter Text")
input_method = st.sidebar.radio("Choose input method:", ["Type Text", "Upload .txt File"])

# Input
texts = []
if input_method == "Type Text":
    text_input = st.text_area("Enter text to analyze sentiment:", height=200)
    if text_input:
        texts = [text_input]
else:
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.splitlines() if line.strip()]

# Process Button
if st.button("Analyze") and texts:
    results = []
    for idx, text in enumerate(texts):
        analysis = nlp(text)[0]
        r = Rake()
        r.extract_keywords_from_text(text)
        keywords = r.get_ranked_phrases()
        results.append({
            "Text": text[:100] + ("..." if len(text) > 100 else ""),
            "Sentiment": analysis["label"],
            "Confidence": round(analysis["score"], 3),
            "Keywords": ", ".join(keywords[:5])
        })

    # Dataframe
    df = pd.DataFrame(results)
    st.subheader("📊 Analysis Results")
    st.dataframe(df)

    # Sentiment Distribution Chart
    st.subheader("📈 Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', color=["green", "red", "gray"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    st.pyplot(fig)

    # Word Cloud of Keywords
    all_keywords = " ".join(df["Keywords"].tolist())
    if all_keywords:
        st.subheader("☁️ Keyword Word Cloud")
        wc = WordCloud(background_color="white", colormap="viridis").generate(all_keywords)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # Download Results
    st.subheader("📥 Download Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")

else:
    st.info("Enter or upload text to begin analysis.")

