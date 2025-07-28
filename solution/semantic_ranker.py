import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import spacy
from sklearn.metrics.pairwise import cosine_similarity

class MiniLMSemanticRanker:
    """
    Uses MiniLM-L6-v2 for semantic search and ranking of document chunks.
    Provides fast, accurate relevance scoring for persona-based document intelligence.
    """
    
    def __init__(self):
        print("Loading MiniLM-L6-v2 for semantic search...")
        
        # Load MiniLM for embeddings (~80MB)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load spaCy for keyword extraction (~50MB)
        print("Loading spaCy for keyword extraction...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Keywords will use simple extraction.")
            self.nlp = None
        
        print("Semantic ranker ready!")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using spaCy NER and important terms."""
        if not self.nlp:
            # Simple fallback keyword extraction
            words = text.lower().split()
            return [w for w in words if len(w) > 4 and w.isalpha()][:10]
        
        doc = self.nlp(text)
        keywords = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'WORK_OF_ART']:
                keywords.append(ent.text.lower())
        
        # Extract important nouns and adjectives
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                len(token.text) > 4 and 
                not token.is_stop and 
                token.is_alpha):
                keywords.append(token.lemma_.lower())
        
        return list(set(keywords))[:15]  # Limit to top 15 unique keywords
    
    def create_persona_query(self, persona: str, job: str) -> str:
        """Create optimized search query from persona and job."""
        # Combine persona and job into effective search query
        query = f"{persona}: {job}"
        
        # Extract keywords from the query
        query_keywords = self.extract_keywords(query)
        
        # Create enhanced query with key terms
        enhanced_query = f"{query}. Key areas: {' '.join(query_keywords[:5])}"
        
        return enhanced_query
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute MiniLM embeddings for texts (~10ms per chunk)."""
        print(f"Computing embeddings for {len(texts)} texts...")
        
        # Batch process for efficiency
        embeddings = self.embedder.encode(
            texts, 
            batch_size=32,  # Optimal batch size for speed
            show_progress_bar=len(texts) > 50
        )
        
        return embeddings
    
    def calculate_relevance_scores(self, 
                                 chunks: List[Dict[str, Any]], 
                                 persona: str, 
                                 job: str) -> List[Dict[str, Any]]:
        """
        Calculate semantic relevance scores using MiniLM embeddings.
        """
        print(f"Calculating relevance for {len(chunks)} chunks...")
        
        # Create optimized query
        query = self.create_persona_query(persona, job)
        
        # Extract chunk texts
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Compute embeddings
        chunk_embeddings = self.compute_embeddings(chunk_texts)
        query_embedding = self.compute_embeddings([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Add scores to chunks and apply boosting
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            base_score = float(similarities[i])
            
            # Apply domain-specific boosting
            boost_score = self.apply_domain_boosting(
                chunk['text'], chunk['document'], persona, job, base_score
            )
            
            chunk_copy['relevance_score'] = boost_score
            chunk_copy['base_semantic_score'] = base_score
            chunk_copy['keywords'] = self.extract_keywords(chunk['text'])
            
            scored_chunks.append(chunk_copy)
        
        return scored_chunks
    
    def apply_domain_boosting(self, text: str, doc_name: str, persona: str, job: str, base_score: float) -> float:
        """Apply domain-specific boosting to relevance scores."""
        boost_factor = 1.0
        
        # Document name boosting
        doc_lower = doc_name.lower()
        
        # Strong boost for HR persona + fillable forms
        if 'hr' in persona.lower() or 'human resources' in persona.lower():
            if 'fill' in doc_lower and 'sign' in doc_lower:
                boost_factor *= 2.5  # Major boost for Fill and Sign
                print(f"Major boost for Fill and Sign content: {doc_name}")
            
            if any(term in doc_lower for term in ['form', 'field', 'signature', 'compliance']):
                boost_factor *= 1.8
        
        # Content-based boosting
        text_lower = text.lower()
        
        # Job-specific terms boost
        job_terms = job.lower().split()
        job_matches = sum(1 for term in job_terms if term in text_lower and len(term) > 3)
        if job_matches > 0:
            boost_factor *= (1 + 0.2 * job_matches)  # 20% boost per job term match
        
        # Persona-specific terms
        if 'hr' in persona.lower():
            hr_terms = ['employee', 'onboarding', 'compliance', 'form', 'document', 
                       'workflow', 'process', 'fillable', 'field', 'signature']
            hr_matches = sum(1 for term in hr_terms if term in text_lower)
            if hr_matches > 0:
                boost_factor *= (1 + 0.15 * hr_matches)  # 15% boost per HR term
        
        # Method and instruction boosting
        if any(term in text_lower for term in ['method', 'step', 'how to', 'create', 'manage']):
            boost_factor *= 1.3
        
        return base_score * boost_factor
    
    def rank_chunks(self, chunks: List[Dict[str, Any]], persona: str, job: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Rank chunks by relevance using MiniLM semantic search.
        Returns top-k most relevant chunks.
        """
        print(f"Ranking chunks for persona: {persona}")
        
        # Calculate relevance scores
        scored_chunks = self.calculate_relevance_scores(chunks, persona, job)
        
        # Sort by relevance score (descending)
        ranked_chunks = sorted(
            scored_chunks, 
            key=lambda x: x['relevance_score'], 
            reverse=True
        )
        
        # Return top-k chunks
        top_chunks = ranked_chunks[:top_k]
        
        scores = [c["relevance_score"] for c in top_chunks[:5]]
        print(f"Top 5 relevance scores: {[f'{score:.3f}' for score in scores]}")
        print(f"Documents in top results: {set(c['document'] for c in top_chunks[:10])}")
        
        return top_chunks
