import os
import requests
import langchain
import autogen
import io
import json

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


#print(get_airtable_records("appbszeQsW74jcj8l","tblspL8JV5EvyDOfa"))
#print(write_single_airtable_record("appbszeQsW74jcj8l","tblspL8JV5EvyDOfa",{'MATERIAL': 'TESTMATERIAL','QUANTITY':'TEST123','PROPERTIES':'PROPS','SOURCE':'TESTSOURCE'}))
#recBJciYxhM0ZojN6
#print(update_single_airtable_record("appbszeQsW74jcj8l","tblspL8JV5EvyDOfa","recBJciYxhM0ZojN6",{'MATERIAL': 'TESTMATERIAL UPDATE','QUANTITY':'TEST123 UPDATE','PROPERTIES':'PROPS UPDATE','SOURCE':'TESTSOURCE UPDATE'}))