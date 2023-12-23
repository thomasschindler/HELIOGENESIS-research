import os
import requests
import langchain
import autogen
import io
import json
import pyairtable

from pyairtable import Api
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
from pypdf import PdfReader

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

# ------------------ Create functions ------------------ #

# Function for google search
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
    print("RESPONSE:", response.text)
    return response.text

# Function for scraping
def summary(objective, content):
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

def web_scraping(objective: str, url: str):
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
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
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")        


# Function for get airtable records
def get_airtable_records(base_id, table_id):
    api = Api(airtable_api_key)
    table = api.table(base_id, table_id)
    return table.all()

# Function for update airtable records

def write_single_airtable_record(base_id, table_id, fields):
    api = Api(airtable_api_key)
    table = api.table(base_id, table_id)
    return table.create(fields)


# Function for update airtable records

def update_single_airtable_record(base_id, table_id, field_id, fields):
    api = Api(airtable_api_key)
    table = api.table(base_id, table_id)
    return table.update(field_id,fields)

# Function to read PDF files from an url
def pdf_read_from_url(url: str):

    print("#########READING PDF")


    r = requests.get(url)
    f = io.BytesIO(r.content)

    reader = PdfReader(f)
    contents = [url]

    for page in reader.pages:
        contents.append(page.extract_text().split('\n'))

    return contents

# ------------------ Create agent ------------------ #

# Create user proxy agent
user_proxy = UserProxyAgent(name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1
    )

# Create researcher agent
researcher = GPTAssistantAgent(
    name = "researcher",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_GvlOjkiEj1glf1hyLieUzMdT"
    }
)

researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search,
        "pdf_read_from_url": pdf_read_from_url,
        "write_single_airtable_record": write_single_airtable_record
    }
)

# Create research manager agent
research_manager = GPTAssistantAgent(
    name="research_manager",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_CRNMBrzfmCB7CGGFPHd0Hb9I"
    }
)

research_manager.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search,
        "pdf_read_from_url": pdf_read_from_url,
        "write_single_airtable_record": write_single_airtable_record
    }
)

# Create director agent
director = GPTAssistantAgent(
    name = "director",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_weMPr0NNk76xu0m5aydL7zrK",
    }
)

director.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search,
        "pdf_read_from_url": pdf_read_from_url,
        "write_single_airtable_record": write_single_airtable_record
    }
)


# Create group chat
groupchat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager, director], messages=[], max_round=15)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


# ------------------ start conversation ------------------ #
message = """
We are commissioning a comprehensive study on heliogenesis, a concept proposing that all materials required for human civilization can be produced through natural processes and made freely accessible to everyone globally.
Do not rely on external databases or institutional access to services, instead rely on searching google and scraping websites.

The first task is to compile an inventory of materials currently utilized by humanity. This inventory will include a list of materials, outlining their key properties, the quantities (in metric tons) in which they are required, and references from existing articles – please ensure all sources are credible and verifiable. The aim is not an exhaustive list of every material in existence, but rather a broad overview capturing the types and quantities of materials essential for our civilization.
Focus on the materials required for the following fields: construction industry, transport and logistics, fashion, furniture, communication.
Examples of the materials we intend to find are concrete, clay, copper, steel, plastic, glass, aluminum, wood, bronze, iron, porcelain, rubber, fabric, etc

Focus your research on running Google searches to find PDF files, then read the contents of those PDF files with the provided function pdf_read_from_url.

Once the data is gathered, it should be organized and stored in an Airtable. Use the following format for the entries: "MATERIAL", "PROPERTIES", "QUANTITY", "SOURCE". The data should be entered into this specific table: https://airtable.com/appbszeQsW74jcj8l/tblspL8JV5EvyDOfa/viwdw2SsAl3wISS6R?blocks=hide Please stick to these four fields only, without adding any extra information.

This study is a critical step in understanding the scope of materials needed for a heliogenic civilization and will serve as a foundation for further research and development in this area.
"""
user_proxy.initiate_chat(group_chat_manager, message=message)

#https://en.wikipedia.org/wiki/List_of_companies_of_Germany