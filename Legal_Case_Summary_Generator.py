import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from pathlib import Path

# Page config
st.set_page_config(page_title="📚 Legal Case Summary Generator", layout="centered")

# Purple theme styling
st.markdown("""
    <style>
        html, body, .main {
            background-color: #3b0a57;
            color: #ffffff;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #f5a623;
            text-align: center;
            margin-bottom: 30px;
        }
        .subtitle {
            font-size: 18px;
            color: #e0d7f9;
            text-align: center;
        }
        .summary-box {
            background-color: #5e239d;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #f5a623;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #f5a623;
            color: #000;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ffcf66;
            color: #000;
        }
        .stSelectbox label {
            color: #ffffff;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📚 Legal Case Summary Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Built with LangChain + FAISS + Ollama</div>', unsafe_allow_html=True)

# Load PDFs
pdf_folder = Path("pdfs")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    st.warning("⚠️ No PDF files found in the 'pdfs/' folder.")
else:
    selected_file = st.selectbox("📂 Select a legal case PDF", [file.name for file in pdf_files])

    if selected_file:
        file_path = pdf_folder / selected_file
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # LLM and Embeddings
        llm = Ollama(model="gemma3:1b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = FAISS.from_documents(chunks, embedding=embeddings)

        # 🔶 Prompt Template: Headings Only Format
        court_prompt_template = """
You are a legal assistant. Read the legal case provided below and generate a structured summary using clear section-wise headings. Write each section in short paragraphs. Avoid bullet points. Use the following heading format:

**Case Summary Format (Use Headings Only):**

1. **Case Title**

2. **Court Name and Date**

3. **Parties Involved**

4. **Facts of the Case**

5. **Legal Issues**

6. **Arguments Presented**

7. **Judgment**

8. **Reasoning of the Court**

9. **Applicable Laws and Precedents**

10. **Final Outcome**

Use these exact headings in your output. Be clear and concise. If any information is missing in the text, skip that section gracefully.

**Text:**  
{text}

Now generate the summary using the above format.
"""
        prompt = PromptTemplate(input_variables=["text"], template=court_prompt_template)

        # Chain
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=True
        )

        if st.button("🧾 Generate Court-Style Summary"):
            with st.spinner("⚖️ Summarizing the legal case..."):
                summary = chain.run(chunks)
                st.success("✅ Summary generated!")

                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.subheader("📄 Legal Case Summary")
                st.markdown(summary)
                st.markdown('</div>', unsafe_allow_html=True)


