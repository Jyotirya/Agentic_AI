"""
Test Suite - Unit tests for the RAG system
This module contains tests for all major components.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.scraper_agent import DocumentScraper, scrape_department_data
from agents.embedding_agent import EmbeddingManager
from agents.query_agent import QueryManager
from utils.chunking import (
    RecursiveChunking, SemanticChunking, FixedSizeChunking,
    chunk_documents, get_chunking_strategy
)
from utils.prompts import classify_query_type, get_prompt_for_query_type


class TestScraperAgent(unittest.TestCase):
    """Test scraper agent functionality"""
    
    def setUp(self):
        self.scraper = DocumentScraper()
    
    def test_scraper_initialization(self):
        """Test scraper initializes correctly"""
        self.assertIsNotNone(self.scraper)
        self.assertTrue(self.scraper.data_dir.exists())
    
    def test_scrape_all(self):
        """Test scraping all documents"""
        docs, metadata = self.scraper.scrape_all()
        self.assertIsInstance(docs, list)
        self.assertIsInstance(metadata, list)
        self.assertEqual(len(docs), len(metadata))
    
    def test_backward_compatibility(self):
        """Test backward compatible function"""
        docs = scrape_department_data()
        self.assertIsInstance(docs, list)


class TestEmbeddingAgent(unittest.TestCase):
    """Test embedding agent functionality"""
    
    def setUp(self):
        self.manager = EmbeddingManager()
    
    def test_manager_initialization(self):
        """Test embedding manager initializes correctly"""
        self.assertIsNotNone(self.manager)
        self.assertIsNotNone(self.manager.model)
    
    def test_embed_batch(self):
        """Test batch embedding"""
        texts = ["Test document 1", "Test document 2"]
        embeddings = self.manager.embed_batch(texts, show_progress=False)
        self.assertEqual(len(embeddings), len(texts))
        self.assertIsInstance(embeddings[0], list)


class TestQueryAgent(unittest.TestCase):
    """Test query agent functionality"""
    
    def setUp(self):
        self.manager = QueryManager(use_reranker=False)
    
    def test_manager_initialization(self):
        """Test query manager initializes correctly"""
        self.assertIsNotNone(self.manager)
        self.assertIsNotNone(self.manager.model)
    
    def test_query_stats(self):
        """Test query statistics"""
        stats = self.manager.get_query_stats("test query")
        self.assertIn("query", stats)
        self.assertIn("query_length", stats)
        self.assertIn("word_count", stats)


class TestChunkingStrategies(unittest.TestCase):
    """Test chunking strategies"""
    
    def setUp(self):
        self.documents = [
            "This is a test document. It has multiple sentences. Each sentence adds information.",
            "Another document here. With different content. For testing purposes."
        ]
    
    def test_recursive_chunking(self):
        """Test recursive chunking strategy"""
        strategy = RecursiveChunking(chunk_size=50, chunk_overlap=10)
        chunks, metadata = strategy.chunk(self.documents)
        self.assertIsInstance(chunks, list)
        self.assertIsInstance(metadata, list)
        self.assertEqual(len(chunks), len(metadata))
    
    def test_semantic_chunking(self):
        """Test semantic chunking strategy"""
        strategy = SemanticChunking(chunk_size=50, chunk_overlap=10)
        chunks, metadata = strategy.chunk(self.documents)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_fixed_size_chunking(self):
        """Test fixed size chunking strategy"""
        strategy = FixedSizeChunking(chunk_size=50, chunk_overlap=10)
        chunks, metadata = strategy.chunk(self.documents)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_backward_compatibility(self):
        """Test backward compatible chunking function"""
        chunks = chunk_documents(self.documents, chunk_size=50)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_strategy_factory(self):
        """Test chunking strategy factory"""
        strategy = get_chunking_strategy("recursive")
        self.assertIsInstance(strategy, RecursiveChunking)


class TestPrompts(unittest.TestCase):
    """Test prompt utilities"""
    
    def test_query_classification(self):
        """Test query type classification"""
        course_query = "What are the prerequisites for EE720?"
        faculty_query = "Who teaches power electronics?"
        research_query = "What research is happening in VLSI?"
        
        self.assertEqual(classify_query_type(course_query), "course")
        self.assertEqual(classify_query_type(faculty_query), "faculty")
        self.assertEqual(classify_query_type(research_query), "research")
    
    def test_prompt_selection(self):
        """Test prompt selection for query types"""
        prompt = get_prompt_for_query_type("course")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_pipeline_simulation(self):
        """Test simulated full pipeline"""
        # Scrape
        scraper = DocumentScraper()
        docs, metadata = scraper.scrape_all()
        
        if not docs:
            self.skipTest("No documents available for testing")
        
        # Chunk
        chunks = chunk_documents(docs[:2], chunk_size=200)
        self.assertGreater(len(chunks), 0)
        
        # Embed (mock)
        manager = EmbeddingManager()
        embeddings = manager.embed_batch(chunks[:5], show_progress=False)
        self.assertEqual(len(embeddings), min(5, len(chunks)))


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
