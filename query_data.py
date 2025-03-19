import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text):
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results.", ""
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    
    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
    sources_text = "\n".join(sources)
    
    return response_text, sources_text

# Create the Gradio UI
demo = gr.Interface(
    fn=query_rag,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=[gr.Textbox(label="Response"), gr.Textbox(label="Sources")],
    title="RAG Chatbot",
    description="Ask questions and get answers based on the retrieved documents from ChromaDB."
)

demo.launch()
