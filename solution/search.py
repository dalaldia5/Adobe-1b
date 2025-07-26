import numpy as np
import re
import random
from collections import Counter
from typing import List, Dict, Any

class GeneralizedSemanticSearcher:
    """
    A generalized semantic searcher that works across different domains.
    Uses keyword matching and TF-IDF style scoring with domain-agnostic patterns.
    """
    def __init__(self):
        # Initialize domain-agnostic keyword categories
        self.keyword_categories = {
            # Academic research keywords
            "research": [
                "research", "study", "analysis", "investigation", "experiment",
                "hypothesis", "theory", "model", "methodology", "findings",
                "literature review", "citation", "reference", "publication"
            ],
            
            # Business and finance keywords
            "business": [
                "revenue", "profit", "market", "strategy", "investment", "growth",
                "performance", "financial", "budget", "cost", "roi", "analysis",
                "competitive advantage", "market share", "customer", "sales"
            ],
            
            # Education and learning keywords
            "education": [
                "learning", "education", "student", "teacher", "course", "curriculum",
                "skill", "knowledge", "concept", "principle", "understanding",
                "assessment", "evaluation", "grade", "academic", "study"
            ],
            
            # Technical and methodology keywords
            "technical": [
                "method", "approach", "technique", "process", "procedure", "algorithm",
                "implementation", "application", "system", "framework", "tool",
                "software", "technology", "innovation", "development"
            ],
            
            # General analysis keywords
            "analysis": [
                "analyze", "evaluate", "assess", "examine", "review", "compare",
                "contrast", "measure", "quantify", "qualify", "interpret",
                "conclude", "summarize", "synthesize", "investigate"
            ]
        }
        
        self.chunks = []
        
    def create_index(self, chunks: list):
        """
        Stores chunks for later search.
        """
        self.chunks = chunks
        print(f"Created simple index with {len(chunks)} chunks")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization by splitting on whitespace and removing punctuation.
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split on whitespace and filter out empty tokens
        tokens = [token for token in text.split() if token]
        
        return tokens
    
    def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Compute term frequency for a list of tokens.
        """
        tf_dict = {}
        counter = Counter(tokens)
        
        for token, count in counter.items():
            tf_dict[token] = count / len(tokens)
            
        return tf_dict
    
    def analyze_query(self, query: str):
        """
        Analyze the query to identify key themes and important terms.
        Returns enhanced query and category weights.
        """
        query_lower = query.lower()
        category_weights = {category: 0 for category in self.keyword_categories}
        
        # Count occurrences of keywords from each category in the query
        for category, keywords in self.keyword_categories.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    category_weights[category] += 1
        
        # Identify the dominant categories (those with non-zero weights)
        dominant_categories = [cat for cat, weight in category_weights.items() if weight > 0]
        
        # If no dominant categories found, assume all are relevant
        if not dominant_categories:
            dominant_categories = list(self.keyword_categories.keys())
        
        # Create enhanced query by adding representative keywords from dominant categories
        enhanced_terms = []
        for category in dominant_categories:
            # Add 2-3 random keywords from each dominant category
            sample_size = min(3, len(self.keyword_categories[category]))
            sampled_keywords = random.sample(self.keyword_categories[category], sample_size)
            enhanced_terms.extend(sampled_keywords)
        
        # Create enhanced query
        enhanced_query = query + " " + " ".join(enhanced_terms)
        
        return enhanced_query, category_weights
    
    def calculate_relevance_score(self, chunk: Dict[str, Any], query_tokens: List[str], category_weights: Dict[str, int]) -> float:
        """
        Calculate a relevance score for a chunk based on its content and the query tokens.
        """
        content = chunk.get('content', '').lower()
        section_title = chunk.get('section_title', '').lower()
        doc_name = chunk.get('doc_name', '').lower()
        
        # Tokenize chunk content
        content_tokens = self.tokenize(content)
        
        # Calculate term frequency for content and query
        content_tf = self.compute_tf(content_tokens)
        
        # Calculate score based on token overlap
        score = 0
        for token in query_tokens:
            if token in content_tf:
                score += content_tf[token] * 10  # Weight by term frequency
        
        # Score based on keyword categories
        for category, weight in category_weights.items():
            # Only consider categories with non-zero weight
            if weight > 0:
                category_score = 0
                for keyword in self.keyword_categories[category]:
                    # Check content
                    if keyword.lower() in content:
                        category_score += 2
                    # Check title (higher weight)
                    if keyword.lower() in section_title:
                        category_score += 3
                    # Check document name
                    if keyword.lower() in doc_name:
                        category_score += 1
                
                # Apply category weight
                score += category_score * weight
        
        # Consider content length (longer content might be more informative)
        words = content.split()
        if len(words) > 100:
            score += 1
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> list:
        """
        Performs search against the indexed chunks using keyword matching.
        """
        if not self.chunks:
            print("Warning: Search index is empty.")
            return []
        
        # Analyze query to enhance it and get category weights
        enhanced_query, category_weights = self.analyze_query(query)
        
        # Tokenize query
        query_tokens = self.tokenize(enhanced_query)
        
        # Calculate relevance scores for each chunk
        result_chunks = []
        for chunk in self.chunks:
            score = self.calculate_relevance_score(chunk, query_tokens, category_weights)
            result_chunk = chunk.copy()
            result_chunk['relevance_score'] = score
            result_chunks.append(result_chunk)
        
        # Sort by relevance score
        result_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Remove duplicate content
        unique_chunks = []
        seen_content = set()
        for chunk in result_chunks:
            # Create a simplified representation of the content for deduplication
            content_hash = hash(chunk.get('content', '')[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        # Return top_k unique results
        return unique_chunks[:top_k]
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Performs simple extractive summarization by selecting the most important sentences.
        """
        # Handle very short text
        if len(text.split()) < 40:
            return text
        
        # Clean the text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = cleaned_text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return cleaned_text
        
        # Score sentences based on importance
        sentence_scores = []
        for sentence in sentences:
            # Base score on length (not too short, not too long)
            length_score = min(len(sentence.split()) / 5, 10)
            
            # Boost score for instructional content
            instruction_score = 0
            if re.search(r'^(To|How to|Steps to|You can)', sentence, re.IGNORECASE):
                instruction_score += 5
            if ":" in sentence:
                instruction_score += 3
            if re.search(r'(use|select|click|choose|open|create|convert|fill)', sentence, re.IGNORECASE):
                instruction_score += 2
                
            # Boost score for important keywords
            keyword_score = 0
            for category, keywords in self.keyword_categories.items():
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        keyword_score += 1
            
            total_score = length_score + instruction_score + keyword_score
            sentence_scores.append((sentence, total_score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:num_sentences]]
        
        # Reorder sentences to maintain original flow
        ordered_sentences = []
        for sentence in sentences:
            if sentence in top_sentences and sentence not in ordered_sentences:
                ordered_sentences.append(sentence)
                if len(ordered_sentences) == num_sentences:
                    break
        
        # If we couldn't get enough sentences, just use the top ones
        if not ordered_sentences:
            ordered_sentences = [s[0] for s in sentence_scores[:num_sentences]]
        
        return '. '.join(ordered_sentences) + '.'

# For backward compatibility
SemanticSearcher = GeneralizedSemanticSearcher
SimpleSemanticSearcher = GeneralizedSemanticSearcher