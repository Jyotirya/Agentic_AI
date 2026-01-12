"""
Response Agent - Generates natural language answers using LLM
This module handles the generation of contextual responses using Google's Gemini API.
"""

import os
import logging
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
        Estimate confidence score based on answer and context
        
        Args:
            answer: Generated answer
            context: Context chunks used
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic: check for uncertainty phrases
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "unclear", 
            "not enough information", "cannot determine",
            "i apologize", "i cannot"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return 0.3
        
        # Check if answer references context
        if "context" in answer_lower or "according to" in answer_lower:
            return 0.85
        
        return 0.7
    
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
