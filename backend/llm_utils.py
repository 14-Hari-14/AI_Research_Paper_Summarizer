from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.3)

def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)