import os
import requests
import io
from pypdf import PdfReader


def pdf_read_from_url(url):

    print("#########READING PDF")


    r = requests.get(url)
    f = io.BytesIO(r.content)

    reader = PdfReader(f)
    contents = [url]

    for page in reader.pages:
        contents.append(page.extract_text().split('\n'))

    return contents


print(pdf_read_from_url("https://zoom.thomas.cr/data/jessie.pdf"))