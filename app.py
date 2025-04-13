import validators
import streamlit as st
from urllib.parse import urlparse, parse_qs
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from fpdf import FPDF


# 🔧 Streamlit Page Setup
st.set_page_config(page_title="🦜 LangChain: All-in-One Summarizer", page_icon="🧠", layout="centered")
st.title("🧠 LangChain: All-in-One Summarizer")
st.caption("Summarize 📹 YouTube videos or 🌐 Web pages using your favorite LLM provider.")

# 🔑 Sidebar Inputs
with st.sidebar:
    st.header("🔑 API Keys")
    provider = st.selectbox("Choose LLM Provider", ["Groq", "OpenAI", "HuggingFace"])
    groq_api_key = st.text_input("Groq API Key", type="password") if provider == "Groq" else None
    openai_api_key = st.text_input("OpenAI API Key", type="password") if provider == "OpenAI" else None
    hf_api_key = st.text_input("HuggingFace API Token", type="password") if provider == "HuggingFace" else None

# 🌐 URL Input
generic_url = st.text_input("Enter YouTube or Website URL", "")

# 🧠 Prompt Template
prompt_template = """
You are a helpful assistant. Please summarize the following content in no more than 300 words:

Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# 🧠 LLM Selection
llm = None
if provider == "Groq" and groq_api_key:
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
elif provider == "OpenAI" and openai_api_key:
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
elif provider == "HuggingFace" and hf_api_key:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_api_key,
        task="text-generation",
        max_length=500,
        temperature=0.7
    )

# 📼 YouTube Transcript Fallback
@st.cache_data(show_spinner=False)
def get_youtube_transcript(url):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path[1:]
    else:
        video_id = parse_qs(parsed.query).get("v", [None])[0]

    if not video_id:
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(entry["text"] for entry in transcript)
    except TranscriptsDisabled:
        # fallback using yt_dlp
        try:
            ydl_opts = {
                "quiet": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "vtt",
                "outtmpl": "%(id)s.%(ext)s"
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                subtitles = info.get("subtitles") or info.get("automatic_captions")
                if subtitles and "en" in subtitles:
                    st.warning("📄 Subtitles available but direct download not implemented.")
                    return None
                else:
                    return None
        except Exception:
            return None


# 📥 Content Loader
def load_content(url):
    if "youtube.com" in url or "youtu.be" in url:
        transcript = get_youtube_transcript(url)
        if transcript:
            return [Document(page_content=transcript)]
        else:
            st.error("❌ Could not extract transcript from the YouTube video.")
            return []
    else:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return loader.load()

# 📤 PDF Downloader
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_path = "/tmp/summary.pdf"
    pdf.output(pdf_path)
    return pdf_path


# 🔘 Main Logic
if st.button("✨ Summarize Now"):
    if not generic_url.strip():
        st.error("❌ Please enter a valid URL.")
    elif not validators.url(generic_url):
        st.error("⚠️ The entered text is not a valid URL.")
    elif not llm:
        st.error("⚠️ Please enter the correct API key for the selected provider.")
    else:
        try:
            with st.spinner("🔄 Loading and summarizing..."):
                docs = load_content(generic_url)
                if not docs:
                    st.stop()

                trimmed_text = docs[0].page_content[:12000]
                st.info(f"📝 Content length: {len(trimmed_text)} characters")

                with st.expander("📖 Preview Fetched Content"):
                    st.write(trimmed_text[:1000] + "..." if len(trimmed_text) > 1000 else trimmed_text)

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run([Document(page_content=trimmed_text)])

                st.success("✅ Summary Generated!")
                st.markdown(summary)

                # 🔽 PDF Download
                pdf_path = generate_pdf(summary)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download Summary as PDF", f, file_name="summary.pdf")

        except Exception as e:
            st.exception(f"❌ Summarization failed: {e}")
