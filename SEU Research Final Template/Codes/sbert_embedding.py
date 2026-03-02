"""
SBERT Embedding Generation for Movie Scripts
Converts long scripts to 384-dimensional embeddings using chunking.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text, chunk_size=256, overlap=50):
    """
    Split long text into overlapping chunks.
    
    Args:
        text: Input text string
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += step
        
        if end >= len(words):
            break
    
    return chunks


def generate_sbert_embedding(text, model_name='all-MiniLM-L6-v2'):
    """
    Generate SBERT embedding with chunking for long texts.
    
    Args:
        text: Input script text
        model_name: SBERT model name (default: all-MiniLM-L6-v2)
    
    Returns:
        numpy array of shape (384,) - averaged embedding
    """
    # Load SBERT model
    model = SentenceTransformer(model_name)
    
    # Chunk the text
    chunks = chunk_text(text)
    
    if not chunks:
        return np.zeros(384)
    
    # Embed all chunks
    chunk_embeddings = model.encode(chunks, show_progress_bar=False)
    
    # Average chunk embeddings
    if len(chunk_embeddings) > 0:
        doc_embedding = np.mean(chunk_embeddings, axis=0)
    else:
        doc_embedding = np.zeros(384)
    
    return doc_embedding


# Example usage
if __name__ == "__main__":
    # Example movie script
    script = """
    [SCENE START]
    JOHN: Hello Mary, how are you today?
    MARY: I'm doing well, thank you for asking.
    JOHN: Would you like to go for a walk?
    MARY: That sounds wonderful! Let's go.
    [SCENE END]
    """
    
    # Generate embedding
    embedding = generate_sbert_embedding(script)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 dimensions: {embedding[:10]}")
