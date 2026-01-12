"""
Scraper Agent - Handles data loading and preprocessing
This module manages loading data from various sources and formatting it for the RAG pipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")

class DocumentScraper:
    """Manages document scraping and preprocessing"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize the document scraper
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.metadata_cache = {}
        
    def scrape_all(self) -> Tuple[List[str], List[Dict]]:
        """
        Scrape all documents from the data directory
        
        Returns:
            Tuple of (documents, metadata) lists
        """
        documents = []
        metadata_list = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return documents, metadata_list
        
        for file_path in self.data_dir.glob("*.json"):
            try:
                docs, meta = self._load_json_file(file_path)
                documents.extend(docs)
                metadata_list.extend(meta)
                logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents, metadata_list
    
    def _load_json_file(self, file_path: Path) -> Tuple[List[str], List[Dict]]:
        """
        Load and process a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Tuple of (documents, metadata) lists
        """
        documents = []
        metadata_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            
        # Handle both list and dict formats
        if isinstance(items, dict):
            items = [items]
        
        for idx, item in enumerate(items):
            doc_text = self._format_item(item)
            documents.append(doc_text)
            
            # Create metadata
            metadata = {
                "source": file_path.name,
                "source_type": self._get_source_type(file_path.name),
                "doc_id": f"{file_path.stem}_{idx}",
                "timestamp": datetime.now().isoformat(),
                **self._extract_metadata(item)
            }
            metadata_list.append(metadata)
        
        return documents, metadata_list
    
    def _format_item(self, item: Dict) -> str:
        """
        Format a data item into a readable document string
        
        Args:
            item: Dictionary containing item data
            
        Returns:
            Formatted document string
        """
        # Special formatting for different data types
        if "course_code" in item:
            return self._format_course(item)
        elif "name" in item and "position" in item:
            return self._format_faculty(item)
        elif "title" in item and "date" in item:
            return self._format_announcement(item)
        else:
            # Generic formatting
            return "\n".join(f"{k}: {v}" for k, v in item.items() if v)
    
    def _format_course(self, course: Dict) -> str:
        """Format course information"""
        parts = [
            f"Course Code: {course.get('course_code', 'N/A')}",
            f"Course Title: {course.get('title', 'N/A')}",
            f"Instructor: {course.get('instructor', 'N/A')}",
        ]
        
        if course.get('prerequisites'):
            parts.append(f"Prerequisites: {course.get('prerequisites')}")
        
        if course.get('description'):
            parts.append(f"Description: {course.get('description')}")
        
        if course.get('credits'):
            parts.append(f"Credits: {course.get('credits')}")
        
        return "\n".join(parts)
    
    def _format_faculty(self, faculty: Dict) -> str:
        """Format faculty information"""
        parts = [
            f"Name: {faculty.get('name', 'N/A')}",
            f"Position: {faculty.get('position', 'N/A')}",
        ]
        
        if faculty.get('department'):
            parts.append(f"Department: {faculty.get('department')}")
        
        if faculty.get('research_interests'):
            interests = faculty.get('research_interests')
            if isinstance(interests, list):
                interests = ", ".join(interests)
            parts.append(f"Research Interests: {interests}")
        
        if faculty.get('email'):
            parts.append(f"Email: {faculty.get('email')}")
        
        if faculty.get('office'):
            parts.append(f"Office: {faculty.get('office')}")
        
        return "\n".join(parts)
    
    def _format_announcement(self, announcement: Dict) -> str:
        """Format announcement information"""
        parts = [
            f"Title: {announcement.get('title', 'N/A')}",
            f"Date: {announcement.get('date', 'N/A')}",
        ]
        
        if announcement.get('category'):
            parts.append(f"Category: {announcement.get('category')}")
        
        if announcement.get('content'):
            parts.append(f"Content: {announcement.get('content')}")
        
        if announcement.get('deadline'):
            parts.append(f"Deadline: {announcement.get('deadline')}")
        
        return "\n".join(parts)
    
    def _get_source_type(self, filename: str) -> str:
        """Determine the type of data based on filename"""
        filename_lower = filename.lower()
        if 'course' in filename_lower:
            return 'course'
        elif 'faculty' in filename_lower:
            return 'faculty'
        elif 'announcement' in filename_lower:
            return 'announcement'
        elif 'research' in filename_lower:
            return 'research'
        else:
            return 'general'
    
    def _extract_metadata(self, item: Dict) -> Dict:
        """Extract relevant metadata from an item"""
        metadata = {}
        
        # Extract common fields
        if 'course_code' in item:
            metadata['course_code'] = item['course_code']
        if 'name' in item:
            metadata['name'] = item['name']
        if 'title' in item:
            metadata['title'] = item['title']
        if 'date' in item:
            metadata['date'] = item['date']
        if 'category' in item:
            metadata['category'] = item['category']
        
        return metadata
    
    def scrape_by_type(self, data_type: str) -> Tuple[List[str], List[Dict]]:
        """
        Scrape documents of a specific type
        
        Args:
            data_type: Type of data to scrape (course, faculty, announcement, etc.)
            
        Returns:
            Tuple of (documents, metadata) lists
        """
        documents = []
        metadata_list = []
        
        for file_path in self.data_dir.glob("*.json"):
            if data_type.lower() in file_path.name.lower():
                try:
                    docs, meta = self._load_json_file(file_path)
                    documents.extend(docs)
                    metadata_list.extend(meta)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents, metadata_list
    
    def validate_data(self) -> Dict[str, any]:
        """
        Validate the data directory and files
        
        Returns:
            Validation report dictionary
        """
        report = {
            "directory_exists": self.data_dir.exists(),
            "files_found": [],
            "total_documents": 0,
            "errors": []
        }
        
        if not self.data_dir.exists():
            report["errors"].append(f"Data directory {self.data_dir} does not exist")
            return report
        
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    count = len(data) if isinstance(data, list) else 1
                    report["files_found"].append({
                        "filename": file_path.name,
                        "document_count": count
                    })
                    report["total_documents"] += count
            except Exception as e:
                report["errors"].append(f"Error in {file_path.name}: {str(e)}")
        
        return report


# Global scraper instance
_scraper = None

def get_scraper() -> DocumentScraper:
    """Get or create scraper instance"""
    global _scraper
    if _scraper is None:
        _scraper = DocumentScraper()
    return _scraper


def scrape_department_data() -> List[str]:
    """
    Scrape all department data (backward compatible)
    
    Returns:
        List of document strings
    """
    scraper = get_scraper()
    documents, _ = scraper.scrape_all()
    return documents


def scrape_department_data_with_metadata() -> Tuple[List[str], List[Dict]]:
    """
    Scrape all department data with metadata
    
    Returns:
        Tuple of (documents, metadata) lists
    """
    scraper = get_scraper()
    return scraper.scrape_all()
