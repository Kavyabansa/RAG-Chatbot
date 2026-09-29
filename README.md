# 🤖 Kavya's ML Journey — RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** that answers questions about Kavya's Machine Learning journey, including weekly learning progress, projects, datasets, model performance, technologies, and key concepts.

The application combines **Streamlit, FAISS, Sentence Transformers, LangChain, and Groq LLMs** to provide context-aware answers based on a curated knowledge base.

## 🚀 Live Demo

🌐 **Try the chatbot:**
https://rag-chatbot-jft4avcfwwbo2sp7pzgw8c.streamlit.app/

Ask questions such as:

* What did I learn in Week 6?
* What accuracy did I achieve on MNIST?
* Which model did I use for transfer learning?
* What NLP projects have I completed?
* What was my Kaggle House Prices score?
* Which tools and frameworks have I learned?
* What did I learn about GenAI?

---

## ✨ Features

* 💬 Interactive chatbot interface
* 🧠 Retrieval-Augmented Generation (RAG)
* 🔎 Semantic document retrieval using embeddings
* ⚡ Fast vector search using FAISS
* 🤗 Sentence Transformer embeddings
* 🔗 LangChain-powered prompt and conversation handling
* 🚀 Groq-powered LLM responses
* 📝 Conversation history and contextual responses
* 📚 Source information included with generated answers
* ☁️ Deployed using Streamlit Community Cloud
* 🔐 API key handled through Streamlit Secrets

---

## 🏗️ System Architecture

```text
                 User Question
                       │
                       ▼
              ┌─────────────────┐
              │   Streamlit UI  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Sentence        │
              │ Transformer     │
              │ Embeddings      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │      FAISS      │
              │  Vector Search  │
              └────────┬────────┘
                       │
                 Top-K Documents
                       │
                       ▼
              ┌─────────────────┐
              │ LangChain Prompt│
              │ + Chat History  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Groq LLM      │
              │ GPT OSS 20B     │
              └────────┬────────┘
                       │
                       ▼
                Generated Answer
```

---

## 🧠 How RAG Works

The chatbot does not simply send every question directly to the LLM.

Instead, it follows a RAG pipeline:

### 1. Knowledge Base

The project contains structured information about:

* Machine Learning
* Deep Learning
* NLP
* Generative AI
* Kaggle projects
* Model performance
* Tools and frameworks
* Weekly learning progress
* Deployed projects

### 2. Embedding Generation

Each knowledge-base document is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

### 3. Vector Database

The generated embeddings are stored in a:

```text
FAISS IndexFlatL2
```

This allows the application to quickly find documents semantically related to the user's question.

### 4. Retrieval

When a user asks a question, the question is converted into an embedding and FAISS retrieves the most relevant documents.

The application retrieves the **top 3 relevant documents**.

### 5. Prompt Construction

The retrieved information is added to the prompt along with the conversation history.

### 6. LLM Generation

The final prompt is sent to the Groq-hosted LLM, which generates the answer using the retrieved context.

---

## 🛠️ Tech Stack

| Technology               | Purpose                       |
| ------------------------ | ----------------------------- |
| 🐍 Python                | Core programming language     |
| 🎈 Streamlit             | Web application interface     |
| 🤗 Sentence Transformers | Text embeddings               |
| 🔎 FAISS                 | Vector similarity search      |
| 🔗 LangChain             | RAG and conversation pipeline |
| ⚡ Groq                   | LLM inference                 |
| 🧠 GPT-OSS-20B           | Language model                |
| 🔢 NumPy                 | Numerical operations          |
| 🔥 PyTorch               | Deep learning backend         |

---

## 📂 Project Structure

```text
kavya-ml-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml.example
```

### `app.py`

Contains the complete Streamlit application, including:

* RAG pipeline
* FAISS index creation
* Sentence Transformer embeddings
* Groq LLM connection
* Prompt construction
* Chat history
* Streamlit interface

### `requirements.txt`

Contains all Python dependencies required to run the application.

### `.streamlit/secrets.toml`

Used locally to store the Groq API key securely.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

The application requires a Groq API key.

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

⚠️ **Never commit your actual API key to GitHub.**

The `.gitignore` file is configured to prevent the secrets file from being uploaded.

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## ☁️ Deployment

This project is deployed using **Streamlit Community Cloud**.

Deployment steps:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application.
4. Select the GitHub repository.
5. Select:

```text
app.py
```

as the main file.
6. Deploy the application.
7. Open **Settings → Secrets**.
8. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

The application can then be accessed through its Streamlit deployment URL.

---

## 📊 Knowledge Base

The chatbot currently contains information covering the following learning journey:

### Week 1–2 — Python for ML

* NumPy
* Pandas
* Matplotlib
* Vectorization
* Broadcasting
* Data manipulation

### Week 3–4 — Machine Learning

* Linear Regression
* Logistic Regression
* Bias-Variance Tradeoff
* Train/Validation/Test Split
* Cross-Validation
* Accuracy
* F1 Score
* ROC-AUC

### Week 5 — Gradient Boosting

* XGBoost
* LightGBM
* Hyperparameter tuning
* RandomizedSearchCV
* Early stopping
* Kaggle House Prices

### Week 6 — Deep Learning

* PyTorch
* Tensors
* Autograd
* CNN
* MNIST
* CIFAR-10
* Batch Normalization
* Dropout

### Week 7 — Transfer Learning

* EfficientNet-B0
* Intel Image Classification
* Fine-tuning
* Grad-CAM
* Explainable AI

### Week 8 — NLP

* Text preprocessing
* TF-IDF
* LSTM
* Transformers
* BERT
* Named Entity Recognition
* Sentiment Analysis

### Week 9 — Generative AI

* Groq API
* Prompt Engineering
* Zero-shot prompting
* Few-shot prompting
* Chain-of-Thought
* LangChain
* RAG
* FAISS
* LoRA
* LLM Agents
* ReAct

---

## 📈 Project Highlights

The knowledge base includes several ML/AI projects and achievements, including:

* 🏠 Kaggle House Prices competition
* 🖼️ Intel Image Classification
* 🧠 MNIST CNN
* 🖼️ CIFAR-10 classification
* 💬 NLP sentiment analysis
* 🏷️ Named Entity Recognition
* 🤖 Generative AI experiments
* 📚 RAG chatbot
* 🚀 Deployed ML/NLP applications

---

## 💡 Example Questions

Try these questions in the chatbot:

```text
What did I learn in Week 5?
```

```text
What was my MNIST accuracy?
```

```text
Which CNN dataset did I work on?
```

```text
What did I learn about BERT?
```

```text
What is RAG according to my Week 9 learning?
```

```text
What was my Kaggle achievement?
```

```text
Which tools have I learned?
```

```text
What projects have I deployed?
```

---

## 🔐 Security

API credentials should **never** be hard-coded into the source code or committed to GitHub.

Use Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

If an API key has previously been exposed publicly, revoke or rotate it before deploying the application.

---

## 🔮 Future Improvements

Possible improvements for future versions:

* 📄 Upload custom PDF/TXT/DOCX files
* 🗂️ Support multiple knowledge bases
* 🔍 Advanced semantic search
* 📊 Retrieval similarity scores
* 📚 Better source citations
* 💾 Persistent vector database
* 👤 User authentication
* 🎨 Custom Streamlit UI
* 🧠 Conversation summarization
* 📈 RAG evaluation using retrieval and generation metrics
* 🔄 Automatic document ingestion pipeline

---

## 👨‍💻 Author

**Kavya Bansal**

B.Tech — Computer Science & Artificial Intelligence

Interested in:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Generative AI
* Large Language Models
* AI Research

---

## 🌐 Live Application

🚀 **RAG Chatbot:**
https://rag-chatbot-jft4avcfwwbo2sp7pzgw8c.streamlit.app/

---

## ⭐ If you found this project interesting

Feel free to explore the project, try the live chatbot, and experiment with the RAG pipeline.

**Built with Python + Streamlit + FAISS + LangChain + Sentence Transformers + Groq 🤖**
