"""
Supervisor - LangGraph-based workflow orchestration
This module manages the RAG pipeline with conditional routing and error handling.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict
import logging
from agents.scraper_agent import scrape_department_data_with_metadata
from agents.embedding_agent import get_embedding_manager
from agents.query_agent import get_query_manager, retrieve_context_with_metadata
from agents.response_agent import generate_answer_with_metadata
from utils.chunking import chunk_documents_with_metadata
from utils.prompts import classify_query_type, get_prompt_for_query_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State object for the graph"""
    # Input
    question: str
    
    # Scraping
    docs: Optional[List[str]]
    doc_metadata: Optional[List[Dict]]
    
    # Chunking
    chunks: Optional[List[str]]
    chunk_metadata: Optional[List[Dict]]
    
    # Embedding
    collection: Optional[any]
    collection_initialized: bool
    
    # Query
    context: Optional[List[str]]
    context_metadata: Optional[List[Dict]]
    context_scores: Optional[List[float]]
    query_type: Optional[str]
    
    # Response
    answer: str
    confidence: Optional[float]
    
    # State management
    error: Optional[str]
    skip_scraping: bool
    retry_count: int


class RAGSupervisor:
    """Manages the RAG workflow"""
    
    def __init__(self, use_persistent_storage: bool = True):
        """
        Initialize the supervisor
        
        Args:
            use_persistent_storage: Whether to use persistent ChromaDB storage
        """
        self.use_persistent_storage = use_persistent_storage
        self.embedding_manager = get_embedding_manager()
        self.query_manager = get_query_manager()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> any:
        """Build the LangGraph workflow"""
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("check_collection", self._check_collection)
        graph.add_node("scrape", self._scrape)
        graph.add_node("chunk", self._chunk)
        graph.add_node("embed", self._embed)
        graph.add_node("classify_query", self._classify_query)
        graph.add_node("query", self._query)
        graph.add_node("respond", self._respond)
        graph.add_node("handle_error", self._handle_error)
        
        # Set entry point
        graph.set_entry_point("check_collection")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "check_collection",
            self._should_initialize,
            {
                "scrape": "scrape",
                "classify": "classify_query"
            }
        )
        
        graph.add_edge("scrape", "chunk")
        graph.add_edge("chunk", "embed")
        graph.add_edge("embed", "classify_query")
        graph.add_edge("classify_query", "query")
        
        graph.add_conditional_edges(
            "query",
            self._check_query_success,
            {
                "respond": "respond",
                "error": "handle_error"
            }
        )
        
        graph.add_edge("respond", END)
        graph.add_edge("handle_error", END)
        
        return graph.compile()
    
    def _check_collection(self, state: GraphState) -> GraphState:
        """Check if collection exists and has data"""
        try:
            collection = self.embedding_manager.create_or_get_collection(reset=False)
            count = collection.count()
            
            logger.info(f"Collection has {count} documents")
            
            state["collection"] = collection
            state["collection_initialized"] = count > 0
            state["skip_scraping"] = count > 0
            
        except Exception as e:
            logger.error(f"Error checking collection: {str(e)}")
            state["collection_initialized"] = False
            state["skip_scraping"] = False
            state["error"] = str(e)
        
        return state
    
    def _should_initialize(self, state: GraphState) -> str:
        """Decide whether to initialize collection or proceed to query"""
        if state.get("collection_initialized", False):
            return "classify"
        else:
            return "scrape"
    
    def _scrape(self, state: GraphState) -> GraphState:
        """Scrape documents from data sources"""
        try:
            logger.info("Scraping department data...")
            docs, metadata = scrape_department_data_with_metadata()
            
            state["docs"] = docs
            state["doc_metadata"] = metadata
            
            logger.info(f"Scraped {len(docs)} documents")
            
        except Exception as e:
            logger.error(f"Error in scraping: {str(e)}")
            state["error"] = f"Scraping error: {str(e)}"
            state["docs"] = []
            state["doc_metadata"] = []
        
        return state
    
    def _chunk(self, state: GraphState) -> GraphState:
        """Chunk documents"""
        try:
            logger.info("Chunking documents...")
            docs = state.get("docs", [])
            metadata = state.get("doc_metadata", [])
            
            if not docs:
                state["chunks"] = []
                state["chunk_metadata"] = []
                return state
            
            chunks, chunk_metadata = chunk_documents_with_metadata(
                docs=docs,
                chunk_size=500,
                chunk_overlap=100,
                metadata_list=metadata,
                strategy="recursive"
            )
            
            state["chunks"] = chunks
            state["chunk_metadata"] = chunk_metadata
            
            logger.info(f"Created {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error in chunking: {str(e)}")
            state["error"] = f"Chunking error: {str(e)}"
            state["chunks"] = []
            state["chunk_metadata"] = []
        
        return state
    
    def _embed(self, state: GraphState) -> GraphState:
        """Embed and store chunks"""
        try:
            logger.info("Embedding and storing chunks...")
            chunks = state.get("chunks", [])
            metadata = state.get("chunk_metadata", [])
            
            if not chunks:
                return state
            
            # Reset collection and add new documents
            self.embedding_manager.create_or_get_collection(reset=True)
            collection = self.embedding_manager.add_documents(
                documents=chunks,
                metadata_list=metadata
            )
            
            state["collection"] = collection
            state["collection_initialized"] = True
            
            logger.info(f"Embedded {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error in embedding: {str(e)}")
            state["error"] = f"Embedding error: {str(e)}"
        
        return state
    
    def _classify_query(self, state: GraphState) -> GraphState:
        """Classify the query type"""
        try:
            question = state.get("question", "")
            query_type = classify_query_type(question)
            
            state["query_type"] = query_type
            logger.info(f"Classified query as: {query_type}")
            
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            state["query_type"] = "general"
        
        return state
    
    def _query(self, state: GraphState) -> GraphState:
        """Query the vector database"""
        try:
            collection = state.get("collection")
            question = state.get("question", "")
            
            if not collection:
                raise ValueError("Collection not initialized")
            
            logger.info(f"Querying for: {question[:50]}...")
            
            # Retrieve context with metadata
            context, metadata, scores = retrieve_context_with_metadata(
                collection=collection,
                query=question,
                k=5
            )
            
            state["context"] = context
            state["context_metadata"] = metadata
            state["context_scores"] = scores
            
            logger.info(f"Retrieved {len(context)} context chunks")
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            state["error"] = f"Query error: {str(e)}"
            state["context"] = []
            state["context_metadata"] = []
            state["context_scores"] = []
        
        return state
    
    def _check_query_success(self, state: GraphState) -> str:
        """Check if query was successful"""
        context = state.get("context", [])
        if context:
            return "respond"
        else:
            return "error"
    
    def _respond(self, state: GraphState) -> GraphState:
        """Generate response"""
        try:
            question = state.get("question", "")
            context = state.get("context", [])
            query_type = state.get("query_type", "general")
            
            logger.info("Generating response...")
            
            # Get appropriate prompt for query type
            system_prompt = get_prompt_for_query_type(query_type)
            
            # Generate answer with metadata
            result = generate_answer_with_metadata(
                question=question,
                context=context,
                system_prompt=system_prompt
            )
            
            state["answer"] = result["answer"]
            state["confidence"] = result.get("confidence", 0.0)
            
            logger.info(f"Generated answer with confidence: {state['confidence']}")
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            state["error"] = f"Response error: {str(e)}"
            state["answer"] = "I apologize, but I encountered an error while generating the response. Please try again."
            state["confidence"] = 0.0
        
        return state
    
    def _handle_error(self, state: GraphState) -> GraphState:
        """Handle errors gracefully"""
        error = state.get("error", "Unknown error")
        logger.error(f"Handling error: {error}")
        
        state["answer"] = f"I apologize, but I encountered an issue: {error}. Please try rephrasing your question or contact support."
        state["confidence"] = 0.0
        
        return state
    
    def invoke(self, input_data: Dict) -> Dict:
        """
        Run the RAG pipeline
        
        Args:
            input_data: Dictionary with 'question' key
            
        Returns:
            Dictionary with 'answer' and other metadata
        """
        # Initialize state
        initial_state = {
            "question": input_data.get("question", ""),
            "docs": None,
            "doc_metadata": None,
            "chunks": None,
            "chunk_metadata": None,
            "collection": None,
            "collection_initialized": False,
            "context": None,
            "context_metadata": None,
            "context_scores": None,
            "query_type": None,
            "answer": "",
            "confidence": None,
            "error": None,
            "skip_scraping": False,
            "retry_count": 0
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result


# Global supervisor instance
_supervisor = None

def get_supervisor() -> RAGSupervisor:
    """Get or create supervisor instance"""
    global _supervisor
    if _supervisor is None:
        _supervisor = RAGSupervisor()
    return _supervisor


def supervisor():
    """Create and return a compiled supervisor graph (backward compatible)"""
    sup = get_supervisor()
    return sup
