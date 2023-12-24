import os
import getpass
import chromadb

from PyPDF2 import PdfReader

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from chromadb import PersistentClient

from pydantic import BaseModel  # Import BaseModel from pydantic

# Initialize the OpenAI language model
model = ChatOpenAI()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],  # Specify the input variables used in the template
    template="{context}\nQuestion: {question}"
)

class SimpleOutput(BaseModel):
    answer: str

output_parser = PydanticOutputParser(pydantic_object=SimpleOutput)
llm_chain = LLMChain(llm=model, prompt=prompt_template, output_parser=output_parser)

# Define a custom retriever that queries the ChromaDB collection
class CustomChromaRetriever:
    def __init__(self, collection):
        self.collection = collection

    def get_relevant_documents(self, query, n_results=10):
        query_embedding = embedding.embed_query(query)
        return self.collection.query(query_embeddings=query_embedding, n_results=n_results)

    def as_retriever(self):
        # Return the object that MultiQueryRetriever expects (this is speculative)
        return self


# Set OpenAI API Key for embeddings and language model
os.environ['OPENAI_API_KEY'] = "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"  # Replace with your actual API key

# Directory containing PDF documents
pdf_directory = '/app/INPUT/'

# Initialize embeddings
embedding = OpenAIEmbeddings()

# Initialize ChromaDB client
chromadb_client = PersistentClient(path='/app/INPUT/CHROMA3/')  # Adjust with your ChromaDB configuration


# Load PDFs and create RAGs and collections
collections = []
rags = []

for filename in os.listdir(pdf_directory):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        page_texts = [page.page_content for page in pages if page.page_content]
        # Generate embeddings for the pages' content
        embeddings = embedding.embed_documents(page_texts)
        # Create a collection for this RAG
        collection_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the collection name
        if any(tmp_collection.name == collection_name for tmp_collection in chromadb_client.list_collections()):
            print("loading collection")
            collection = chromadb_client.get_collection(name=collection_name)
        else:
            collection = chromadb_client.create_collection(name=collection_name)

            for page_embedding, page in zip(embeddings, pages):
                # Add the embedding and metadata to the collection
                collection.add(
                    ids=[f"{collection_name}_{page.metadata.get('page')}"],
                    embeddings=[page_embedding],
                    metadatas=[{"source": pdf_path, "page": page.metadata.get("page")}]
                )

        collections.append(collection)
        rags.append(model)


# Choose one collection for the retriever (you can modify this to use multiple collections)
# Initialize MultiQueryRetriever with the custom ChromaDB retriever
#retriever = MultiQueryRetriever(retriever=CustomChromaRetriever(collections[0]), llm_chain=llm_chain)
# Assuming llm_chain is already defined as in your current implementation
retriever = MultiQueryRetriever.from_llm(
    retriever=CustomChromaRetriever(collections[0]).as_retriever(),
    llm_chain=llm_chain
)


# User-provided question
user_question = "What are the latest advancements in AI?"

# Let RAGs enter a conversation
responses = []
for rag in rags:
    # Embed the user question
    embedded_query = embedding.embed_query(user_question)
    docs = retriever.get_relevant_documents(query=embedded_query, n_results=5)
    context = " ".join([doc.page_content for doc in docs])
    response = rag.generate(f"{context}\nQuestion: {user_question}")
    responses.append(response)

# Generate a structured summary from the conversation
structured_summary = {
    "question": user_question,
    "responses": [],
    "sources": []
}

for response, doc in zip(responses, docs):
    structured_summary["responses"].append(response)
    structured_summary["sources"].append({"path": doc.metadata["source"], "page": doc.metadata["page"]})

# Print the structured summary
print(structured_summary)
