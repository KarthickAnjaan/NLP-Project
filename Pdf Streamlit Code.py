import pandas as pd
import numpy as np
import PyPDF2
from PyPDF2 import PdfReader
from pdfreader import SimplePDFViewer
import cv2
import pytesseract
from IPython.display import Image
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit as st 
import pdfplumber
import pickle

symmetric_embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
b= st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


a= st.file_uploader("Upload pdf", type=['pdf'])

if a:

    # Getting query as User input
    text2 = st.text_input("Query", "Search for a sentence")


    if st.button("Predict"):
            
               pdf_reader = PyPDF2.PdfReader(a)
               text_1=''
               for i in range(0,3):
                 page = pdf_reader.pages[i]
                 text_1+= page.extract_text()
    
    # Get the number of pages in the PDF file
               #num_pages = pdf_reader.getNumPages()
    
    # Loop over all pages and extract the text
               #for page_num in range(num_pages):
        # Get the current page
                 #page = pdf_reader.getPage(page_num)
                 #page_text = page.extractText()

                       
               sentences = nltk.sent_tokenize(text_1)
         #Encoding sentences
               corpus_1 = symmetric_embedder.encode(sentences)

         # Encoding the Query input
               query_embedding = symmetric_embedder.encode(text2, convert_to_tensor=True)
         # Performing Semantic search on corpus and query:
               hits = util.semantic_search(query_embedding, corpus_1)
               hits = hits[0]
               hits1 = [{'corpus_id': sentences[hit['corpus_id']], 'score': round(hit['score'], 4)} for hit in hits]
               df = pd.DataFrame(hits1)
               st.write(df)

         
