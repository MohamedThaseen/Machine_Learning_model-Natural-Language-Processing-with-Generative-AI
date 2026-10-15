# Medical Assistant – RAG-Based Clinical Q&A System

A Retrieval-Augmented Generation (RAG) AI prototype that lets healthcare professionals query trusted medical references (the Merck Manuals) in natural language and get fast, grounded, evidence-based answers — instead of manually searching a 4,000+ page document.

## Business Context

The healthcare industry is rapidly evolving, and professionals face increasing challenges in managing vast volumes of medical data while delivering accurate and timely diagnoses. Quick access to comprehensive, reliable, and up-to-date medical knowledge is critical for improving patient outcomes and ensuring informed decision-making in a fast-paced environment.

Healthcare professionals often encounter information overload, struggling to sift through extensive research and data to create accurate diagnoses and treatment plans. This challenge is amplified by the need for efficiency, particularly in emergencies, where time-sensitive decisions are vital. Furthermore, access to trusted, current medical information from renowned manuals and research papers is essential for maintaining high standards of care.

To address these challenges, healthcare centers can focus on integrating systems that streamline access to medical knowledge, provide tools to support quick decision-making, and enhance efficiency. Leveraging centralized knowledge platforms and ensuring healthcare providers have continuous access to reliable resources can significantly improve patient care and operational effectiveness.

## Objective

As an AI specialist, the task is to develop a RAG-based AI solution using renowned medical manuals to address healthcare challenges — understanding information overload, applying AI techniques to streamline decision-making, analyzing its impact on diagnostics and patient outcomes, evaluating its potential to standardize care practices, and building a functional prototype demonstrating feasibility and effectiveness.

### Questions the prototype answers

1. What is the protocol for managing sepsis in a critical care unit?
2. What are the common symptoms of appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?
3. What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?
4. What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?
5. What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?

## Dataset

**The Merck Manuals** — medical references published by Merck & Co. since 1899, covering a wide range of medical topics including disorders, tests, diagnoses, and drugs.

- Source: `medical_diagnosis_manual.pdf`
- Over 4,000 pages across 23 sections

## Approach

1. **Baseline: LLM-only Q&A (no retrieval)** — the five clinical questions were first posed directly to open-source LLMs (no document grounding) to establish a baseline and confirm the need for retrieval-augmented context.
2. **Model Selection** – three open-source, quantized (GGUF) LLMs were benchmarked for response quality and latency:
   - `Llama-2-13B-chat-GGUF` (13B parameters)
   - `Llama-2-7B-chat-GGUF` (7B parameters)
   - `Mistral-7B-Instruct-v0.2-GGUF` (7B parameters) — **selected**, as it answered queries fastest with comparable quality.
3. **Prompt Engineering** – a structured system/user prompt template was applied to the chosen LLM to improve answer quality before adding retrieval.
4. **RAG Pipeline — Data Preparation**:
   - Loaded the PDF manual with `PyMuPDFLoader`
   - Chunked the text with `RecursiveCharacterTextSplitter` (chunk size 512 tokens, 25-token overlap, `cl100k_base` encoding)
   - Generated embeddings using the `thenlper/gte-large` sentence-transformer model
   - Indexed chunks in a **Chroma** vector database
   - Built a similarity-search retriever (top-k relevant chunks) to supply grounded context to the LLM
5. **RAG-based Q&A** – the five clinical questions were re-run through the full RAG pipeline (retrieved context + Mistral-7B-Instruct) and compared against the no-retrieval baseline.
6. **Fine-Tuned Generation** – response parameters (temperature, top-p, top-k, context size `k`) were tuned to improve answer quality and specificity.
7. **Output Evaluation** – an LLM-as-judge approach scored each RAG answer on two custom rubrics (1–5 scale):
   - **Groundedness** – is the answer derived only from the retrieved context?
   - **Relevance** – does the answer address the key aspects of the question?

## Results

- The **Mistral-7B-Instruct-v0.2 (Q8_0 GGUF)** model was chosen as the best-performing LLM — fastest response time (under 3 minutes per query) among the three models tested, with comparable answer quality.
- Adding retrieval (RAG) produced noticeably more specific, manual-grounded answers compared to the LLM-only baseline, which tended to answer from general pretrained knowledge rather than the source manual.
- On the custom **relevance** rubric, the RAG pipeline scored **4 out of 5** on 3 of the 5 evaluation questions, indicating strong — though not perfect — alignment between answers and the retrieved context.
- The Mistral model required a context window of at least 1,800 tokens; a window of 2,048 tokens was used to safely accommodate the retrieved context plus prompt overhead.

## Business Recommendations

- **Develop Clinical Decision Support Tools** – integrate RAG with LLMs to offer real-time diagnostic and treatment support that gives physicians evidence-based recommendations, improving patient outcomes and satisfaction.
- **Patient-Facing Virtual Assistants** – build secure, privacy-compliant AI chatbots powered by RAG to give patients accurate, contextual answers to medical queries, reducing staff workload and improving patient engagement.
- **Data Integration and Security** – prioritize integration with electronic health records and up-to-date medical databases while ensuring compliance with privacy regulations such as HIPAA, since safeguarding sensitive health data is critical.
- **Custom Model Fine-Tuning** – fine-tune LLMs on medical reports, doctor-patient conversations, and clinical guidelines relevant to the organization's specialty to improve accuracy and contextual relevance.

## Tech Stack

- **Python**: pandas, tiktoken
- **LLM Runtime**: `llama-cpp-python` (GGUF quantized models via Hugging Face Hub)
- **LLMs evaluated**: Llama-2-13B-chat, Llama-2-7B-chat, Mistral-7B-Instruct-v0.2
- **RAG Framework**: LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-classic`)
- **PDF Parsing**: PyMuPDF (`PyMuPDFLoader`)
- **Embeddings**: Sentence-Transformers (`thenlper/gte-large`)
- **Vector Database**: ChromaDB
- **Evaluation**: LLM-as-judge (custom groundedness and relevance rubrics)

## Repository Structure

```
.
├── Natural_Language_Processing_with_Generative_AI__Medical_Assistant.ipynb   # Full RAG pipeline: baseline, model selection, chunking, embedding, retrieval, evaluation
├── medical_diagnosis_manual.pdf                                             # Source manual (not included — see Data Access below)
└── README.md
```

## Data Access

The Merck Manual PDF used as the knowledge source is not included in this repository due to size and licensing. Place `medical_diagnosis_manual.pdf` in the project directory (or update the file path in the notebook) before running.

## How to Run

1. Clone the repository and install the dependencies:
   ```bash
   pip install pymupdf langchain langchain-community langchain-core langchain-text-splitters langchain-classic chromadb sentence-transformers huggingface_hub tiktoken pandas llama-cpp-python
   ```
2. Place `medical_diagnosis_manual.pdf` in the project directory.
3. Open and run the notebook in Jupyter Notebook, JupyterLab, or Google Colab (a GPU runtime is recommended for faster LLM inference; a CPU-only install path is also provided in the notebook).

## Disclaimer

This is an academic/prototype project. It is **not** a validated clinical tool and should not be used to make real medical decisions without review by qualified healthcare professionals.

## Author

Mohamed Rifakath Ali A
