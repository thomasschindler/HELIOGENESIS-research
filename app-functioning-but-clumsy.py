import os
import requests
import langchain
import autogen
import io
import json
import pyairtable
import chromadb
import hashlib
import uuid

from textblob import TextBlob
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyairtable import Api
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain.chat_models import ChatOpenAI


browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
airtable_base_id = os.getenv("AIRTABLE_BASE_ID")
airtable_table_rag_id = os.getenv("AIRTABLE_TABLE_RAG_ID")
airtable_table_question_id = os.getenv("AIRTABLE_TABLE_QUESTION_ID")
airtable_table_response_id = os.getenv("AIRTABLE_TABLE_RESPONSE_ID")
airtable_table_final_id = os.getenv("AIRTABLE_TABLE_FINAL_ID")


input_dir = "/app/INPUT/"
chroma_dir = input_dir + "CHROMA"
client = OpenAI()


# make chroma persistent
if not os.path.exists(chroma_dir):
    os.makedirs(chroma_dir)

chromadb_client = chromadb.PersistentClient(path=chroma_dir)
#chromadb_client = chromadb.Client({'storage': {'path': chroma_dir}})

# these will have to come from a database
query_text = "How could humans produce materials based on natural processes?"
query_statement = "nature based materials"

PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ------------------ Create functions ------------------ #

def google_search(search_keyword):    
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    return response.text

def text_summarize(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = langchain.prompts.PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

def web_scrape(objective: str, url: str):
    
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url        
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(f"https://chrome.browserless.io/content?token={browserless_api_key}", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        
        if len(text) > 10000:
            output = text_summarize(objective,text)
            return output
        else:
            return text
    else:
        empty_return = true
        #print(f"HTTP request failed with status code {response.status_code}")        

def airtable_records_get(base_id, table_id):
    api = Api(airtable_api_key)
    table = api.table(base_id, table_id)
    return table.all()

def airtable_record_write(base_id, table_id, fields):
    api = Api(airtable_api_key)
    table = api.table(base_id, table_id)
    return table.create(fields)

def airtable_record_update(base_id, table_id, field_id, fields):
    api = Api(airtable_api_key)
    table = api.table(base_id, table_id)
    return table.update(field_id,fields)

def pdf_from_url_read(url: str):
    r = requests.get(url)
    f = io.BytesIO(r.content)
    reader = PdfReader(f)
    contents = [url]
    for page in reader.pages:
        contents.append(page.extract_text().split('\n'))
    return contents

def pdf_from_path_read(pdf_path):
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = []
            # Read each page from the PDF
            for page in reader.pages:
                text.append(page.extract_text())

            
            return ' '.join(text)
    except Exception as e:
        #print(f"Could not read {pdf_path}: {e}")
        return None

def scan_directory_for_pdfs(directory):
    records = []
    if not os.path.isdir(directory):
        #print(f"The directory {directory} does not exist.")
        return records

    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            hash_md5 = hashlib.md5(pdf_path.encode()).hexdigest()
            records.append({
                'fields': {
                    'PDF': pdf_path,
                    'NAME': hash_md5
                }
            })
    return records

def input_to_token_limit_trim(input_text, max_tokens):
    # Tokenize the input
    tokens = client.completions.create(prompt=input_text, model="gpt-3.5-turbo-instruct").usage.total_tokens
    # Check if the number of tokens exceeds the limit
    if tokens > max_tokens:
        # Split the text and reduce it to fit the token limit
        words = input_text.split()
        while tokens > max_tokens:
            words.pop(0)  # Remove words from the beginning
            tokens = client.completions.create(prompt=' '.join(words), model="gpt-3.5-turbo-instruct").usage.total_tokens
        input_text = ' '.join(words)

    return input_text

def sentiment_negative(response_text):
    analysis = TextBlob(response_text)
    # You might adjust these thresholds based on what works best for your model
    if analysis.sentiment.polarity < 0.1 and analysis.sentiment.subjectivity < 0.4:
        return True  # The sentiment suggests no clear answer was provided
    return False


#BUILD THE RAGS
#rag_pdf = airtable_records_get(airtable_base_id,airtable_table_rag_id)
rag_pdf = scan_directory_for_pdfs(input_dir)


# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
)
# List to hold collections
# we might use this if we want the RAGs to critique the final result
# not for now
#collections_list = []





# Process each record
collections = chromadb_client.list_collections()
for record in rag_pdf:
    #if we already have a collection, skip this.
    if any(tmp_collection.name == record['fields']['NAME'] for tmp_collection in collections):
        #print(f"loading collection")        
        collection = chromadb_client.get_collection(record['fields']['NAME'])
    else:
        #print(f"reading PDF {record['fields']['PDF']} ")
        contents = pdf_from_path_read(record['fields']['PDF'])
        #print(f"content read")
        collection = chromadb_client.create_collection(name=record['fields']['NAME'])
        # Flatten the list of lists into a single list of strings (lines)
        all_lines = [line for page in contents[1:] for line in page]
        # Join the lines to form a single text block
        full_text = '\n'.join(all_lines)
        # Split the document into chunks
        try:
            #print(f"chunking")
            chunks = text_splitter.split_text(full_text)
            chunk_ids = []
            for chunk in chunks:
                chunk_ids.append(str(uuid.uuid4()))

            collection.add(
                documents=chunks,
                ids=chunk_ids
                )
            #print(f"collection built")        
        except AttributeError as e:
            nullreturn = true
            #print(f"An error occurred: {e}")

    # now we get the relevant chunks from the collection
    results = collection.query(
        query_texts=[query_statement],
        n_results=5
        )
    #print(f"results collected")
    context_text = ""
    if 'documents' in results and results['documents']:
        # Iterate over each document in the results
        for document_set in results['documents']:
            for document in document_set:
                # Append each document's text to the all_documents_string variable
                context_text += document + "\n\n"  # Add a space between documents for readability
        context_text = input_to_token_limit_trim(input_text=context_text, max_tokens=3000)
    else:
        nullreturn = true
        #print("No documents found or 'documents' key not in results.")
    # then we construct the query
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # get a resonse from an LLM
    try:
        #print(f"connecting to LLM")
        model = ChatOpenAI()
        response_text = model.predict(prompt)
        if sentiment_negative(response_text) is False:
            print(response_text)
            # and finally store the response to airtable
    except AttributeError as e:
        nullreturn = true
        #print(f"An error occurred: {e}")



