class embedding_models:
    def embedding_googlepalm():
        
        embeddings=GooglePalmEmbeddings(model_name="models/embedding-gecko-001",google_api_key=os.getenv("GOOGLE_API_KEY"))

    def embedding_GoogleGenerativeAIEmbeddings():
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def embedding_googlevertexai():
        from vertexai.language_models import TextEmbeddingModel
        model= model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

class chat_models:
    def Chat_GoogleGenerativeAI():
        from langchain_google_genai import ChatGoogleGenerativeAI
        ChatGoogleGenerativeAI(model="gemini-pro",
                                temperature=0.7,
                                google_api_key="A")