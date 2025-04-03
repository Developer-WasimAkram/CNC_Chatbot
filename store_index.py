from src.utils import load_pdf_files,text_split,download_hugging_face_embedding
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = "pcsk_6d3uNY_NfSiPAVzwopGGrK1RWhw1RFyLu5gWPvXBtS4gHGTr6UAgDTwbgMZ7MWYs99DExZ"

extracted_data=load_pdf_files(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embedding()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cnc"
pc.create_index(name=index_name,dimension=1536, metric="cosine", spec=ServerlessSpec(
        cloud="aws", region="us-east-1" )) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(documents=text_chunks,index_name=index_name, embedding=embeddings, )
