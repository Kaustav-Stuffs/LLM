#!/usr/bin/env python3
"""
Test script to validate document processor fixes with Ollama models.
This script tests the enhanced document processor with various Ollama models.
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.document_processor import DocumentProcessor
from utils.llm_handler import LLMHandler

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_document(content: str, filename: str) -> str:
    """Create a temporary test document."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name

def test_ollama_models():
    """Test document processor with available Ollama models."""
    
    # Initialize components
    llm_handler = LLMHandler()
    document_processor = DocumentProcessor()
    
    # Get available Ollama models
    available_models = llm_handler.get_available_ollama_models()
    
    if not available_models:
        logger.error("No Ollama models found. Please install Ollama and pull some models.")
        logger.info("Example: ollama pull llama2 or ollama pull mistral")
        return False
    
    logger.info(f"Found {len(available_models)} Ollama models: {available_models}")
    
    # Test content - mix of technical and general information
    test_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. There are three main types of machine learning:
    
    1. Supervised Learning: Uses labeled data to train models. Examples include classification and regression tasks.
    
    2. Unsupervised Learning: Finds patterns in data without labels. Common techniques include clustering and dimensionality reduction.
    
    3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties.
    
    Key algorithms include:
    - Linear Regression: Predicts continuous values
    - Decision Trees: Creates tree-like models for decisions
    - Neural Networks: Mimics brain structure for complex pattern recognition
    
    To get started with machine learning:
    1. Learn Python programming
    2. Understand statistics and mathematics
    3. Practice with datasets
    4. Use libraries like scikit-learn, TensorFlow, or PyTorch
    """
    
    # Test with the first available model
    test_model = available_models[0]
    logger.info(f"Testing with model: {test_model}")
    
    try:
        # Configure Ollama with optimized settings
        success = llm_handler.configure_ollama(
            test_model,
            temperature=0.3,
            max_tokens=4000
        )
        
        if not success:
            logger.error(f"Failed to configure Ollama model: {test_model}")
            return False
        
        logger.info(f"Successfully configured {test_model}")
        
        # Create test document
        test_file = create_test_document(test_content, "ml_fundamentals.txt")
        
        try:
            # Process the document
            logger.info("Processing test document...")
            results = document_processor.process_file(
                test_file, 
                "ml_fundamentals.txt",
                llm_handler=llm_handler
            )
            
            # Validate results
            if not results:
                logger.error("No results returned from document processor")
                return False
            
            logger.info(f"Successfully processed document into {len(results)} structured items")
            
            # Display results
            print("\n" + "="*80)
            print("DOCUMENT PROCESSING RESULTS")
            print("="*80)
            
            for i, item in enumerate(results, 1):
                print(f"\n--- Item {i} ---")
                print(f"ID: {item.get('id', 'N/A')}")
                print(f"Text: {item.get('text', 'N/A')[:200]}...")
                
                metadata = item.get('metadata', {})
                print(f"Category: {metadata.get('category', 'N/A')}")
                print(f"Keywords: {', '.join(metadata.get('keywords', []))}")
                print(f"Source: {metadata.get('source', 'N/A')}")
            
            # Validate JSON structure
            try:
                json_str = json.dumps(results, indent=2)
                logger.info("✅ Results are valid JSON")
            except Exception as e:
                logger.error(f"❌ Results are not valid JSON: {e}")
                return False
            
            # Check for required fields
            valid_items = 0
            for item in results:
                if (isinstance(item, dict) and 
                    item.get('id') and 
                    item.get('text') and 
                    item.get('metadata')):
                    valid_items += 1
            
            logger.info(f"✅ {valid_items}/{len(results)} items have valid structure")
            
            if valid_items == len(results):
                logger.info("🎉 ALL TESTS PASSED! Document processor works correctly with Ollama.")
                return True
            else:
                logger.warning(f"⚠️ Some items have invalid structure ({valid_items}/{len(results)})")
                return False
                
        finally:
            # Clean up test file
            os.unlink(test_file)
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

def test_json_recovery():
    """Test the JSON recovery mechanisms with malformed JSON."""
    
    document_processor = DocumentProcessor()
    
    # Test cases with malformed JSON
    test_cases = [
        # Missing comma
        '[{"text": "Q: Test? A: Answer", "metadata": {"category": "Test"}} {"text": "Q: Test2? A: Answer2", "metadata": {"category": "Test"}}]',
        
        # Trailing comma
        '[{"text": "Q: Test? A: Answer", "metadata": {"category": "Test", "keywords": ["test",]}}]',
        
        # Unescaped quotes
        '[{"text": "Q: What\'s this? A: It\'s a test", "metadata": {"category": "Test"}}]',
        
        # Mixed valid/invalid objects
        '[{"text": "Q: Valid? A: Yes", "metadata": {"category": "Test"}}, {invalid object}, {"text": "Q: Also valid? A: Yes", "metadata": {"category": "Test"}}]'
    ]
    
    logger.info("Testing JSON recovery mechanisms...")
    
    for i, test_json in enumerate(test_cases, 1):
        logger.info(f"Testing case {i}: {test_json[:50]}...")
        
        try:
            recovered = document_processor._recover_json_objects(test_json, f"test_case_{i}")
            logger.info(f"✅ Case {i}: Recovered {len(recovered)} objects")
        except Exception as e:
            logger.error(f"❌ Case {i}: Recovery failed - {e}")
    
    logger.info("JSON recovery tests completed")

if __name__ == "__main__":
    print("🧪 Testing Document Processor with Ollama Models")
    print("="*60)
    
    # Test JSON recovery first
    test_json_recovery()
    print()
    
    # Test with Ollama models
    success = test_ollama_models()
    
    if success:
        print("\n🎉 All tests passed! The document processor is now compatible with Ollama models.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        sys.exit(1)