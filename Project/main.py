"""
Main Entry Point - Run the RAG system
This script provides a CLI interface to interact with the RAG system.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from graph.supervisor import get_supervisor
from agents.scraper_agent import get_scraper
from agents.embedding_agent import get_embedding_manager
from config import Config


def initialize_system(reset: bool = False):
    """
    Initialize the RAG system
    
    Args:
        reset: Whether to reset the vector database
    """
    print("🚀 Initializing RAG System...")
    print(f"   Using LLM: {Config.LLM_MODEL}")
    print(f"   Using Embeddings: {Config.EMBEDDING_MODEL}")
    
    # Initialize embedding manager
    embedding_manager = get_embedding_manager()
    
    # Create or get collection
    collection = embedding_manager.create_or_get_collection(reset=reset)
    count = collection.count()
    
    if count == 0:
        print("\n📥 No documents found. Loading data...")
        
        # Load documents
        scraper = get_scraper()
        docs, metadata = scraper.scrape_all()
        
        if not docs:
            print("❌ No documents found in data directory!")
            return False
        
        print(f"   Loaded {len(docs)} documents")
        
        # Chunk and embed
        from utils.chunking import chunk_documents_with_metadata
        chunks, chunk_metadata = chunk_documents_with_metadata(
            docs=docs,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            metadata_list=metadata,
            strategy=Config.CHUNKING_STRATEGY
        )
        
        print(f"   Created {len(chunks)} chunks")
        
        # Store in database
        embedding_manager.add_documents(chunks, metadata_list=chunk_metadata)
        print(f"   ✅ Stored {len(chunks)} chunks in vector database")
    else:
        print(f"   ✅ Found {count} existing documents in database")
    
    print("\n✨ System ready!\n")
    return True


def interactive_mode():
    """Run in interactive CLI mode"""
    print("=" * 60)
    print("IITB EE Department RAG Assistant - Interactive Mode")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop")
    print("Type 'citations' after a query to see detailed sources")
    print("Type 'verify' after a query to see full verification report\n")
    
    supervisor = get_supervisor()
    last_result = None
    
    while True:
        try:
            question = input("❓ You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Handle post-query commands
            if question.lower() == 'citations' and last_result:
                print("\n" + supervisor.format_citations(last_result, "detailed"))
                continue
            
            if question.lower() == 'verify' and last_result:
                print("\n" + supervisor.get_verification_summary(last_result))
                # Show conflict info if any
                conflict_report = last_result.get('conflict_report', {})
                if conflict_report and not conflict_report.get('conflict_free', True):
                    conflicts = conflict_report.get('conflicts', [])
                    print(f"\n⚠️ {len(conflicts)} conflict(s) detected:")
                    for i, conflict in enumerate(conflicts[:3], 1):
                        print(f"  {i}. {conflict.get('description', 'Unknown conflict')}")
                continue
            
            print("\n🤔 Thinking...\n")
            
            result = supervisor.invoke({"question": question})
            last_result = result
            
            print(f"💡 Assistant: {result['answer']}\n")
            
            # Display verification summary
            if result.get('verification_status'):
                print(supervisor.get_verification_summary(result))
            elif result.get('confidence'):
                print(f"   Confidence: {result['confidence']*100:.1f}%")
            
            if result.get('context'):
                print(f"   Sources used: {len(result['context'])}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def single_query_mode(question: str):
    """Process a single query and exit"""
    supervisor = get_supervisor()
    
    print(f"\n❓ Question: {question}")
    print("🤔 Processing...\n")
    
    result = supervisor.invoke({"question": question})
    
    print("=" * 60)
    print("💡 Answer:")
    print("=" * 60)
    print(result['answer'])
    print()
    
    # Display verification summary
    if result.get('verification_status'):
        print(supervisor.get_verification_summary(result))
        
        # Show risk warning if needed
        risk_level = result.get('risk_level', 'unknown')
        if risk_level in ['high', 'critical']:
            print(f"\n🚨 WARNING: {risk_level.upper()} RISK - Verify this information from official sources")
    elif result.get('confidence'):
        print(f"Confidence: {result['confidence']*100:.1f}%")
    
    if result.get('citations'):
        citations = result['citations']
        num_sources = len(set(c.get('source_file', '') for c in citations))
        print(f"\n📚 Sources used: {len(citations)} citations from {num_sources} source(s)")
        
        print("\nTop sources:")
        for i, citation in enumerate(citations[:3], 1):
            source = citation.get('source_file', 'unknown')
            snippet = citation.get('content_snippet', '')[:150]
            relevance = citation.get('relevance_score', 0)
            print(f"\n[{i}] {source} (relevance: {relevance:.0%})")
            print(f"    \"{snippet}...\"")
    elif result.get('context'):
        print(f"Sources used: {len(result['context'])}")
        print("\nTop sources:")
        for i, ctx in enumerate(result['context'][:2], 1):
            print(f"\n[{i}] {ctx[:200]}...")
    
    # Show conflict warnings if any
    conflict_report = result.get('conflict_report', {})
    if conflict_report and not conflict_report.get('conflict_free', True):
        conflicts = conflict_report.get('conflicts', [])
        print(f"\n⚠️ {len(conflicts)} potential conflict(s) detected in sources:")
        for i, conflict in enumerate(conflicts[:2], 1):
            print(f"  {i}. {conflict.get('conflict_type', 'unknown')}: {conflict.get('description', 'No description')}")
    
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IITB EE Department RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode
  python main.py --query "Prerequisites for EE720?"
  python main.py --init --reset            # Reset and rebuild database
  python main.py --web                     # Launch web interface
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize the system'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset the vector database'
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='Launch web interface'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Print configuration'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    if args.config:
        Config.print_config()
        return
    
    # Launch web interface
    if args.web:
        import subprocess
        print("🌐 Launching web interface...")
        subprocess.run(['streamlit', 'run', 'app.py'])
        return
    
    # Initialize system
    if args.init or args.reset:
        initialize_system(reset=args.reset)
        if not args.query:
            return
    
    # Validate configuration
    if not Config.validate():
        print("\n❌ Configuration validation failed. Please check your .env file.")
        return
    
    # Single query mode
    if args.query:
        if not initialize_system(reset=False):
            return
        single_query_mode(args.query)
    else:
        # Interactive mode
        if not initialize_system(reset=False):
            return
        interactive_mode()


if __name__ == "__main__":
    main()
