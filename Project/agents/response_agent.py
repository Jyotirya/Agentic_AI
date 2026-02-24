"""
Response Agent - Generates natural language answers using LLM
This module handles the generation of contextual responses using Google's Gemini API.
Includes enhanced confidence estimation and citation-aware response generation.
"""

import os
import logging
import re
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

class ResponseGenerator:
    """Manages response generation with LLM"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the response generator
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        
    def generate_with_context(
        self,
        question: str,
        context: List[str],
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, any]:
        """
        Generate response using context and question
        
        Args:
            question: User's question
            context: Retrieved context chunks
            system_prompt: System instruction for the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Format context
            context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(context)])
            
            # Create the prompt
            full_prompt = f"""{system_prompt}

Context Information:
{context_text}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, acknowledge this clearly."""

            # Generate response
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            answer = response.text
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "context_used": len(context)
            })
            
            return {
                "answer": answer,
                "context_used": context,
                "confidence": self._estimate_confidence(answer, context),
                "tokens_used": len(full_prompt.split()) + len(answer.split())
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while generating the response. Please try again.",
                "context_used": context,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _estimate_confidence(self, answer: str, context: List[str]) -> float:
        """
        Estimate confidence score based on answer and context.
        Uses multiple heuristics for a more accurate estimate.
        
        Args:
            answer: Generated answer
            context: Context chunks used
            
        Returns:
            Confidence score between 0 and 1
        """
        # Start with base confidence
        confidence = 0.5
        
        # 1. Check for uncertainty phrases (negative impact)
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "unclear", 
            "not enough information", "cannot determine",
            "i apologize", "i cannot", "no information",
            "not found", "not available", "outside the scope"
        ]
        
        answer_lower = answer.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
        if uncertainty_count > 0:
            confidence -= min(0.3, uncertainty_count * 0.1)
        
        # 2. Check for confident language (positive impact)
        confident_phrases = [
            "according to", "based on the", "the context shows",
            "specifically", "it states that", "as mentioned",
            "the information indicates", "clearly"
        ]
        confident_count = sum(1 for phrase in confident_phrases if phrase in answer_lower)
        if confident_count > 0:
            confidence += min(0.2, confident_count * 0.05)
        
        # 3. Check context coverage
        if context:
            # Count how many context terms appear in answer
            context_terms = set()
            for ctx in context:
                context_terms.update(ctx.lower().split())
            
            answer_terms = set(answer_lower.split())
            
            # Remove common words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                        'and', 'or', 'but', 'if', 'then', 'it', 'this', 'that'}
            context_terms -= stopwords
            answer_terms -= stopwords
            
            if answer_terms:
                coverage = len(answer_terms & context_terms) / len(answer_terms)
                confidence += coverage * 0.15
        
        # 4. Check for specific details (positive impact)
        has_numbers = bool(re.search(r'\d+', answer))
        has_codes = bool(re.search(r'[A-Z]{2,4}\d{3,4}', answer))
        has_names = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', answer))
        
        specificity_bonus = (has_numbers * 0.05) + (has_codes * 0.05) + (has_names * 0.05)
        confidence += specificity_bonus
        
        # 5. Answer length heuristic
        answer_words = len(answer.split())
        if answer_words < 10:
            confidence -= 0.1  # Very short answers may be incomplete
        elif answer_words > 50:
            confidence += 0.05  # Longer detailed answers
        
        # 6. Context count bonus
        if len(context) >= 3:
            confidence += 0.1  # Multiple supporting contexts
        
        # Clamp confidence between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Global instance
_response_generator = None

def get_response_generator() -> ResponseGenerator:
    """Get or create response generator instance"""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator


def generate_answer(
    question: str,
    context: List[str],
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate an answer for the given question and context
    
    Args:
        question: User's question
        context: Retrieved context chunks
        system_prompt: Optional custom system prompt
        
    Returns:
        Generated answer string
    """
    from utils.prompts import SYSTEM_PROMPT
    
    generator = get_response_generator()
    prompt = system_prompt or SYSTEM_PROMPT
    
    result = generator.generate_with_context(
        question=question,
        context=context,
        system_prompt=prompt
    )
    
    return result["answer"]


def generate_answer_with_metadata(
    question: str,
    context: List[str],
    system_prompt: Optional[str] = None
) -> Dict[str, any]:
    """
    Generate an answer with full metadata
    
    Args:
        question: User's question
        context: Retrieved context chunks
        system_prompt: Optional custom system prompt
        
    Returns:
        Dictionary with answer and metadata
    """
    from utils.prompts import SYSTEM_PROMPT
    
    generator = get_response_generator()
    prompt = system_prompt or SYSTEM_PROMPT
    
    return generator.generate_with_context(
        question=question,
        context=context,
        system_prompt=prompt
    )
