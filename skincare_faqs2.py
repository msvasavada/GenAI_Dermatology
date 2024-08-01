import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import Settings
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# URL to be used for scraping
DATA_URL = "https://www.advancedderm.com/for-patients/faq"

# Initialize the Gemini LLM
llm = Gemini()

# Initialize the Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone_client.Index("dermatology")

# Initialize BeautifulSoupWebReader
web_reader = BeautifulSoupWebReader()
documents = web_reader.load_data(urls=[DATA_URL])

# Define the new embedding model
embed_model = GeminiEmbedding(model_name="models/gemini-1.5-flash")

# Set global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Create a PineconeVectorStore using the Pinecone index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a StorageContext using the PineconeVectorStore
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Convert the index to a query engine
query_engine = index.as_query_engine()

# Query the index
gemini_response = query_engine.query("What are the warnings of skin cancer?")

print(gemini_response)
