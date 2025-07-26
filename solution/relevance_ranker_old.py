import numpy as np
import re
from typing import List, Dict, Any, Tuple
import logging
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaAwareRelevanceRanker:
    """
    Generalized relevance ranking and summarization using BART and semantic similarity.
    Works across different domains (academic, business, education, etc.) and personas.
    """
    def __init__(self, summarization_model="facebook/bart-large-cnn", embedding_model="all-MiniLM-L6-v2"):
        logger.info("Initializing PersonaAwareRelevanceRanker...")
        
        # Initialize BART for summarization
        logger.info(f"Loading BART model: {summarization_model}")
        self.summarizer = pipeline(
            "summarization",
            model=summarization_model,
            device=-1  # Use CPU
        )
        
        # Initialize sentence transformer for semantic similarity
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Generalized persona patterns for different domains
        self.persona_patterns = {
            "researcher": ["researcher", "phd", "scientist", "academic", "research", "study", "analysis"],
            "student": ["student", "undergraduate", "graduate", "learner", "studying", "exam", "course"],
            "analyst": ["analyst", "investment", "financial", "business", "market", "revenue", "performance"],
            "manager": ["manager", "director", "lead", "supervisor", "executive", "management"],
            "developer": ["developer", "engineer", "programmer", "technical", "software", "coding"],
            "consultant": ["consultant", "advisor", "expert", "specialist", "professional"],
            "teacher": ["teacher", "instructor", "educator", "professor", "faculty"],
            "journalist": ["journalist", "reporter", "writer", "editor", "media", "news"]
        }
        
        # Task-oriented keywords that are domain-agnostic
        self.task_patterns = {
            "review": ["review", "analyze", "evaluate", "assess", "examine", "critique"],
            "summarize": ["summarize", "overview", "summary", "brief", "digest", "synopsis"],
            "compare": ["compare", "contrast", "difference", "similarity", "benchmark", "versus"],
            "learn": ["learn", "understand", "study", "grasp", "comprehend", "master"],
            "implement": ["implement", "apply", "execute", "deploy", "use", "utilize"],
            "research": ["research", "investigate", "explore", "discover", "find", "identify"],
            "prepare": ["prepare", "ready", "setup", "organize", "plan", "arrange"],
            "create": ["create", "build", "develop", "design", "construct", "generate"]
        }
    
    
    def identify_persona_type(self, persona: str) -> str:
        """
        Identify the persona type from the persona description using semantic matching.
        """
        persona_lower = persona.lower()
        
        # Calculate similarity scores for each persona type
        persona_scores = {}
        for persona_type, keywords in self.persona_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in persona_lower:
                    score += 1
            persona_scores[persona_type] = score
        
        # Return the persona type with highest score, or 'researcher' as default
        if not persona_scores or max(persona_scores.values()) == 0:
            return "researcher"  # Default fallback
        
        return max(persona_scores.items(), key=lambda x: x[1])[0]
    
    def identify_task_type(self, job: str) -> str:
        """
        Identify the task type from the job description.
        """
        job_lower = job.lower()
        
        # Calculate similarity scores for each task type
        task_scores = {}
        for task_type, keywords in self.task_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in job_lower:
                    score += 1
            task_scores[task_type] = score
        
        # Return the task type with highest score, or 'research' as default
        if not task_scores or max(task_scores.values()) == 0:
            return "research"  # Default fallback
        
        return max(task_scores.items(), key=lambda x: x[1])[0]
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using sentence transformers.
        """
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            return max(0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_relevance_score(self, chunk: Dict[str, Any], persona: str, job: str) -> float:
        """
        Calculate a comprehensive relevance score using multiple factors:
        1. Semantic similarity with persona and job
        2. Keyword matching
        3. Content type analysis
        4. Section importance
        """
        content = chunk.get('content', '')
        section_title = chunk.get('section_title', '')
        doc_name = chunk.get('doc_name', '')
        
        if not content:
            return 0.0
        
        # Identify persona and task types
        persona_type = self.identify_persona_type(persona)
        task_type = self.identify_task_type(job)
        
        # Create enhanced query from persona and job
        enhanced_query = f"{persona} {job} {persona_type} {task_type}"
        
        # 1. Semantic similarity score (40% weight)
        content_similarity = self.calculate_semantic_similarity(content, enhanced_query)
        title_similarity = self.calculate_semantic_similarity(section_title, enhanced_query) if section_title else 0
        
        semantic_score = (content_similarity * 0.7 + title_similarity * 0.3) * 0.4
        
        # 2. Keyword matching score (30% weight)
        keyword_score = 0
        persona_keywords = self.persona_patterns.get(persona_type, [])
        task_keywords = self.task_patterns.get(task_type, [])
        
        content_lower = content.lower()
        title_lower = section_title.lower()
        
        # Count persona keyword matches
        for keyword in persona_keywords:
            if keyword in content_lower:
                keyword_score += 0.02
            if keyword in title_lower:
                keyword_score += 0.05
        
        # Count task keyword matches
        for keyword in task_keywords:
            if keyword in content_lower:
                keyword_score += 0.03
            if keyword in title_lower:
                keyword_score += 0.06
        
        keyword_score = min(keyword_score, 0.3)  # Cap at 30%
        
        # 3. Content quality score (20% weight)
        content_quality = 0
        
        # Length factor (not too short, not too long)
        word_count = len(content.split())
        if 50 <= word_count <= 500:
            content_quality += 0.1
        elif 20 <= word_count < 50 or 500 < word_count <= 1000:
            content_quality += 0.05
        
        # Structure indicators
        if section_title:
            content_quality += 0.05
        if re.search(r'\d+\.|\-|\•', content):  # Lists or numbered items
            content_quality += 0.03
        if re.search(r'(method|approach|result|conclusion|introduction)', content_lower):
            content_quality += 0.02
        
        content_quality = min(content_quality, 0.2)  # Cap at 20%
        
        # 4. Section importance score (10% weight)
        section_importance = 0
        if section_title:
            # Important section patterns
            important_patterns = [
                r'(abstract|summary|conclusion|result|method|approach|introduction)',
                r'(finding|analysis|discussion|recommendation|implementation)',
                r'(background|literature|review|comparison|evaluation)'
            ]
            for pattern in important_patterns:
                if re.search(pattern, title_lower):
                    section_importance += 0.03
        
        section_importance = min(section_importance, 0.1)  # Cap at 10%
        
        # Combine all scores
        total_score = semantic_score + keyword_score + content_quality + section_importance
        
        return min(total_score, 1.0)  # Ensure score doesn't exceed 1.0
    
    def rank_by_relevance(self, chunks: List[Dict[str, Any]], persona: str, job: str) -> List[Dict[str, Any]]:
        """
        Rank document chunks by relevance to persona and job description.
        """
        if not chunks:
            return []
        
        logger.info(f"Ranking {len(chunks)} chunks for persona: {persona}")
        
        # Calculate relevance scores
        for chunk in chunks:
            chunk['relevance_score'] = self.calculate_relevance_score(chunk, persona, job)
        
        # Sort by relevance score (highest first)
        ranked_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Top 5 relevance scores: {[chunk.get('relevance_score', 0) for chunk in ranked_chunks[:5]]}")
        
        return ranked_chunks
    
    def summarize_with_bart(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """
        Generate a summary using BART model.
        """
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            # Clean the text
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            # Use BART for summarization
            summary = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return summary[0]['summary_text']
        
        except Exception as e:
            logger.warning(f"Error in BART summarization: {e}")
            # Fallback to extractive summarization
            return self._extractive_summarization(text, max_sentences=3)
    
    def _extractive_summarization(self, text: str, max_sentences: int = 3) -> str:
        """
        Fallback extractive summarization method.
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by length and position
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Word count
            if i < len(sentences) * 0.3:  # Early sentences get bonus
                score += 10
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    def extract_sections_for_persona(self, chunks: List[Dict[str, Any]], persona: str, job: str, 
                                   top_sections: int = 10, top_subsections: int = 15) -> Dict[str, Any]:
        """
        Extract and rank the most relevant sections and subsections for a specific persona and job.
        """
        logger.info(f"Extracting sections for persona: {persona}, job: {job}")
        
        # Rank all chunks by relevance
        ranked_chunks = self.rank_by_relevance(chunks, persona, job)
        
        # Group chunks by document and section
        sections_by_doc = {}
        for chunk in ranked_chunks:
            doc_name = chunk.get('doc_name', 'Unknown Document')
            section_title = chunk.get('section_title', 'Untitled Section')
            page_num = chunk.get('page_number', 0)
            
            if doc_name not in sections_by_doc:
                sections_by_doc[doc_name] = {}
            
            if section_title not in sections_by_doc[doc_name]:
                sections_by_doc[doc_name][section_title] = {
                    'chunks': [],
                    'max_relevance': 0,
                    'page_number': page_num
                }
            
            sections_by_doc[doc_name][section_title]['chunks'].append(chunk)
            sections_by_doc[doc_name][section_title]['max_relevance'] = max(
                sections_by_doc[doc_name][section_title]['max_relevance'],
                chunk.get('relevance_score', 0)
            )
        
        # Create ranked sections list
        all_sections = []
        for doc_name, sections in sections_by_doc.items():
            for section_title, section_data in sections.items():
                all_sections.append({
                    'document': doc_name,
                    'section_title': section_title,
                    'page_number': section_data['page_number'],
                    'importance_rank': section_data['max_relevance'],
                    'chunks': section_data['chunks']
                })
        
        # Sort sections by importance
        all_sections.sort(key=lambda x: x['importance_rank'], reverse=True)
        
        # Take top sections
        top_section_list = all_sections[:top_sections]
        
        # Create subsections from top chunks
        top_chunks = ranked_chunks[:top_subsections]
        subsections = []
        
        for i, chunk in enumerate(top_chunks):
            # Generate summary using BART
            content = chunk.get('content', '')
            refined_text = self.summarize_with_bart(content, max_length=100, min_length=20)
            
            subsections.append({
                'document': chunk.get('doc_name', 'Unknown Document'),
                'page_number': chunk.get('page_number', 0),
                'refined_text': refined_text,
                'relevance_score': chunk.get('relevance_score', 0)
            })
        
        return {
            'sections': top_section_list,
            'subsections': subsections
        }

# For backward compatibility
RelevanceRanker = PersonaAwareRelevanceRanker
                    return instructional_text
        
        return ""
    
    def summarize(self, text: str) -> str:
        """
        Generate a summary optimized for instructional content.
        Uses pattern matching and extractive summarization.
        """
        if not text or len(text) < 50:
            return text
            
        # Try to extract instructional content first
        instructional_content = self.extract_instructional_content(text)
        if instructional_content:
            return instructional_content
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Score sentences based on informativeness
            sentence_scores = []
            for sentence in sentences:
                # Base score on length
                score = min(len(sentence.split()) / 5, 10)  # Length score (max 10)
                
                # Boost score for instructional content
                if re.search(r'^(To|How to|Steps to|You can)', sentence, re.IGNORECASE):
                    score += 5
                if ":" in sentence:
                    score += 3
                if re.search(r'(use|select|click|choose|open|create|convert|fill)', sentence, re.IGNORECASE):
                    score += 2
                    
                sentence_scores.append((sentence, score))
            
            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in sentence_scores[:3]]
            
            # Reorder sentences to maintain original flow
            ordered_sentences = []
            for sentence in sentences:
                if sentence in top_sentences and sentence not in ordered_sentences:
                    ordered_sentences.append(sentence)
                    if len(ordered_sentences) == 3:
                        break
            
            return ' '.join(ordered_sentences)
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            # Fallback to first few sentences
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
        
    def extract_section_title(self, text: str) -> str:
        """
        Extract a meaningful section title from text.
        Focuses on action-oriented phrases and key concepts.
        """
        if not text:
            return ""
            
        # Look for action-oriented titles
        title_patterns = [
            r"(Change\s+\w+\s+(?:forms|PDFs)\s+to\s+\w+(?:\s+\w+)?)",
            r"(Create\s+\w+\s+PDFs\s+from\s+\w+\s+files)",
            r"(Convert\s+\w+\s+(?:content|text|data)\s+to\s+PDF)",
            r"(Fill\s+and\s+sign\s+PDF\s+forms)",
            r"(Send\s+a\s+document\s+to\s+get\s+signatures(?:\s+\w+)?)",
            r"(Interactive\s+forms)",
            r"(Flat\s+forms)",
            r"(Form\s+fields)",
            r"(Electronic\s+signatures)",
            r"(PDF\s+conversion)"
        ]
        
        # Try each pattern
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first match and clean it up
                title = matches[0].strip()
                # Capitalize first letter of each word
                title = ' '.join(word.capitalize() for word in title.split())
                if len(title) > 5 and len(title) < 60:
                    return title
        
        # If no patterns match, use the first sentence if it's short enough
        first_sentence = text.split('.')[0].strip()
        if len(first_sentence) < 60 and len(first_sentence.split()) <= 10:
            return first_sentence
            
        return "" 