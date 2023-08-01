import os
import PyPDF2
import nltk
from gensim.models import Word2Vec
from weaviate.client import Client
from langchain.document_loaders import PDFMinerLoader

nltk.download("punkt")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
    return text


def generate_embeddings(text):
    # Tokenize the text into sentences and words
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Train the Word2Vec model
    model = Word2Vec(words, vector_size=100, window=5, min_count=1, workers=4)
    return model


def store_embeddings_in_weaviate(model, pdf_id):
    client = Client("http://localhost:8080")
    # Create an object with the PDF ID as the class name
    class_name = f"PDF_{pdf_id}"
    client.schema.create_object(class_name)
    # Store the embeddings in Weaviate
    for word, vector in model.wv.vocab.items():
        client.data_object.create(
            class_name,
            {
                "name": word,
                "vector": vector.tolist()
            }
        )


def process_pdfs(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(pdf_path)
            model = generate_embeddings(text)
            store_embeddings_in_weaviate(model, os.path.splitext(filename)[0])

# Replace 'your_pdf_directory_path' with the actual path where your PDF files are stored
process_pdfs('your_pdf_directory_path')

