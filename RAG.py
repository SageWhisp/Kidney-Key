import os
import warnings
from together import Together
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import glob
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Initialize Together AI client
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable is not set! Please create a .env file with your API key.")
print('TOGETHER_API_KEY:', TOGETHER_API_KEY)
client = Together(api_key=TOGETHER_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
)

# Global variables for storing embeddings and documents
documents = []
filenames = []
index = None

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def initialize_rag():
    """Initialize the RAG system by loading and indexing documents."""
    global index
    
    # Get all PDF files from knowledge_base directory
    pdf_files = glob.glob("Knowledge_base/*.pdf")
    
    if not pdf_files:
        raise ValueError("No PDF files found in Knowledge_base directory!")
    
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        content = extract_text_from_pdf(pdf_path)
        
        if content:
            # Split content into chunks (roughly 1000 characters each)
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk)
                    # Add chunk number to filename for better tracking
                    filenames.append(f"{filename} (chunk {i+1})")
    
    if not documents:
        raise ValueError("No valid content extracted from PDFs!")
    
    print(f"Successfully processed {len(documents)} chunks from {len(pdf_files)} PDFs")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = embedding_model.encode(documents)
    
    # Set up FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print("RAG system initialized successfully!")

def answer_question(query: str) -> str:
    """
    Answer a question using the RAG system.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The generated answer
    """
    global index
    
    # Initialize RAG system if not already done
    if index is None:
        initialize_rag()
    
    # Get query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Get top similar documents
    scores, indices = index.search(query_embedding, min(3, len(documents)))
    
    # Build context from retrieved documents
    context_parts = []
    relevant_docs = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            doc_info = {
                "content": documents[idx],
                "filename": filenames[idx],
                "score": float(score),
            }
            relevant_docs.append(doc_info)
            context_parts.append(f"[{doc_info['filename']}]\n{doc_info['content']}")
    
    if not relevant_docs:
        return "I couldn't find any relevant information to answer your question."
    
    # Combine context
    context = "\n\n".join(context_parts)
    
    # Create prompt for LLM
    llm_prompt = f"""Answer the question based on the provided context documents.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the information in the context
- If the context doesn't contain enough information, say so
- Mention which document(s) you're referencing
- Start with According to [document name]
- Add brackets to the document name

Answer:"""
    
    try:
        # Generate answer using Together AI
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        
        # Add source information
        sources_list = [doc["filename"] for doc in relevant_docs]
        sources_text = sources_list[0]
        full_answer = f"{answer}\n\n📄 Source Used: {sources_text}"
        
        return full_answer
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"
