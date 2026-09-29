import os
import time
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Kavya's ML Journey Chatbot",
    page_icon="🤖",
    layout="centered",
)

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

documents = [
    "Week 1-2: Python for ML — NumPy arrays, vectorization, broadcasting. Pandas: read CSV, groupby, merge, apply. Matplotlib basics.",
    "Week 3-4: Core ML concepts — Linear regression, logistic regression from scratch. Bias-variance tradeoff. Train/val/test split. Cross-validation. Metrics: accuracy, F1, AUC-ROC.",
    "Week 5: Gradient boosting — XGBoost and LightGBM. Early stopping, hyperparameter tuning with RandomizedSearchCV. Kaggle House Prices score: 0.12933, top 25%.",
    "Week 6: Deep Learning with PyTorch — Tensors, autograd, nn.Module. CNN on MNIST: 99.27% accuracy. CIFAR-10 from scratch: 85.07% accuracy. BatchNorm, Dropout.",
    "Week 7: Transfer Learning — EfficientNet-B0 fine-tuned on Intel Image Classification: 93.27% accuracy. Grad-CAM heatmaps for explainability. Custom Dataset class.",
    "Week 8: NLP — Text preprocessing, TF-IDF 87.59% on IMDB. LSTM 84.33%. Mini Transformer 84.51%. BERT fine-tuning. NER with BERT pipeline. Gradio NLP app deployed.",
    "Week 9: GenAI — Groq API, prompt engineering (zero-shot, few-shot, CoT). LangChain basics, conversation memory. RAG with FAISS. LoRA fine-tuning GPT-2. LLM agents.",
    "Tools and frameworks: Python, PyTorch, scikit-learn, HuggingFace Transformers, LangChain, Groq API, Google Colab T4 GPU, FAISS, sentence-transformers.",
    "Kaggle achievements: House Prices competition top 25%, XGBoost score 0.12933, LightGBM score ~0.13. Submitted multiple models.",
    "Projects deployed: Intel Image Classifier Gradio app (93.27% accuracy), Multi-task NLP app (sentiment + NER). RAG chatbot over ML journey notes.",
    "Key ML concepts learned: bias-variance tradeoff, cross-validation, feature engineering, ensemble methods, stacking, SHAP values for explainability.",
    "Deep learning concepts: backpropagation, gradient descent, activation functions (ReLU, Sigmoid, Tanh), optimizers (Adam, RMSprop, SGD), learning rate scheduling.",
    "NLP concepts: tokenization, stopwords, stemming, BoW, TF-IDF, Word2Vec embeddings, attention mechanism, BERT pretraining (MLM + NSP), fine-tuning.",
    "GenAI concepts: tokens, temperature, context window, prompt engineering, RAG pipeline, LoRA adapters, LLM agents, ReAct pattern, tool use.",
    "Student: Kavya, B.Tech CSE-AI at NIET Greater Noida, AKTU affiliated, 5th semester. Self-directed ML learning journey started from basics to GenAI.",
]

# -----------------------------
# Cached model/index loading
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model and FAISS index...")
def load_retriever():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    doc_embeddings = embedder.encode(documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings.astype("float32"))
    return embedder, index

@st.cache_resource(show_spinner="Connecting to Groq...")
def load_llm(api_key):
    return ChatGroq(
        api_key=api_key,
        model=MODEL_NAME,
        temperature=0.3,
        max_tokens=500,
    )

# -----------------------------
# API key
# -----------------------------
api_key = None

try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key is missing.")
    st.info(
        "For Streamlit Cloud: open your app → Settings → Secrets and add "
        "`GROQ_API_KEY = \"your_key_here\"`."
    )
    st.stop()

embedder, index = load_retriever()
llm = load_llm(api_key)

# -----------------------------
# RAG functions
# -----------------------------
def retrieve(question, top_k=3):
    question_embedding = embedder.encode([question])
    distances, indices = index.search(
        question_embedding.astype("float32"), top_k
    )
    chunks = [documents[i] for i in indices[0]]
    return chunks

def rag_chatbot(question, history):
    chunks = retrieve(question, top_k=3)
    context = "\n".join(chunks)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a helpful assistant that answers questions
about Kavya's ML learning journey.

Use the following context to answer questions:

{context}

If the answer is not in the context, say so honestly.
Always cite which week or topic your answer comes from."""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "history": history,
        "question": question,
    })

    sources = "\n\n**Sources:**\n" + "\n".join(
        [f"• {chunk[:120]}..." for chunk in chunks]
    )

    return response + sources

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 Kavya's ML Journey Chatbot")
st.caption(
    "Ask questions about Kavya's ML learning journey — projects, scores, "
    "concepts, tools, and weekly progress."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses Retrieval-Augmented Generation (RAG), "
        "FAISS, Sentence Transformers, LangChain, and Groq."
    )

    st.subheader("Try asking")
    st.write("• What was my best Kaggle score?")
    st.write("• What accuracy did I get on MNIST?")
    st.write("• Which transfer learning model did I use?")
    st.write("• What NLP tasks did I complete?")
    st.write("• What tools and frameworks did I learn?")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Kavya's ML journey..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert Streamlit messages to LangChain message objects.
    history = []
    for message in st.session_state.messages[:-1]:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chatbot(prompt, history)
            time.sleep(0.2)
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })
