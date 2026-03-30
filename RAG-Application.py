!pip install groq
import os
os.environ["GROQ_API_KEY"] = "pass your groq_api_key here"
client = Groq(api_key="GROQ_API_KEY")

#Groq is an AI company that developed the Language Processing Unit (LPU), a specialized chip designed specifically for high-speed AI inference, not training. It offers significantly lower latency and faster response times for Large Language Models (LLMs) compared to GPUs, enabling real-time generative AI applications.
from groq import Groq   #just checking the working of the models in GROQ
client = Groq()
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
    

#doing the necessary installations
!pip install gradio
!pip install groq
!pip install langchain
!pip install langchain-community
!pip install sentence-transformers
!pip install faiss-cpu
!pip install pypdf
!pip install faiss-gpu-cu12
!pip install langchain_text_splitters
!pip install -U langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu pypdf groq gradio

from google.colab import userdata
GROQ_API_KEY=userdata.get('rag-based')

import os
import gradio as gr
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
vector_db = None

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def process_document(pdf_file):

    global vector_db  #The vector_db variable is declared as global because its value needs to be accessible and modifiable by multiple functions (process_document and ask_question) throughout the application's lifecycle.Without global, vector_db would be a local variable within process_document, and its value would be lost after the function execution, making it inaccessible to ask_question.


    if pdf_file is None:
        return "Please upload a PDF Document first."
    try:

        # Load PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()



        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
           )
        chunks = text_splitter.split_documents(documents)

        # Create FAISS vector database
        vector_db = FAISS.from_documents(
            chunks,
            embedding_model
          )

        return f"Document processed successfully. {len(chunks)} chunks of your document created.Now, proceed to ask your question ahead"

    except Exception as e:
        return f"Error processing document: {str(e)}"


# Question Answering Function
def ask_question(question):

    global vector_db   #The vector_db variable is declared as global because its value needs to be accessible and modifiable by multiple functions (process_document and ask_question) throughout the application's lifecycle.Without global, vector_db would be a local variable within process_document, and its value would be lost after the function execution, making it inaccessible to ask_question.

    if vector_db is None:
        return "Please upload and process a PDF document first."

    try:

        # Retrieve relevant chunks
        docs = vector_db.similarity_search(question, k=4)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant. Answer the question ONLY Using the following context.
If the answer is not in the context, say "I could not find the answer in the provided context."

Context:
{context}

Question:
{question}

Answer clearly and based only on the provided context.
"""
        # Groq LLM call
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )

        response = chat_completion.choices[0].message.content

        return response

    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Gradio Interface (UI interface)
with gr.Blocks() as demo:

    gr.Markdown("# 📄 PDF Document Assistant Developed by Shahid Mazumdar")

    gr.Markdown(
        "Upload a PDF document and ask Your Queries or questions about its content."
    )

    pdf_input = gr.File(label="Upload your PDF Document")

    process_button = gr.Button("Click to Process Document")

    process_output = gr.Textbox(label="Processing Status")

    process_button.click(
        process_document,
        inputs=pdf_input,
        outputs=process_output
    )

    gr.Markdown("## Ask Questions")

    question_input = gr.Textbox(
        label="Enter your question"
    )

    ask_button = gr.Button("Click to Proceed")

    answer_output = gr.Textbox(
        label="Answer",
        lines=10
    )

    ask_button.click(
        ask_question,
        inputs=question_input,
        outputs=answer_output
    )

demo.launch()

