from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    #embeddings = BedrockEmbeddings(
        #credentials_profile_name="default", region_name="us-east-1")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #embeddings = OllamaEmbeddings(model="mistral")
    #embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings
