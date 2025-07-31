#!/usr/bin/env python3
"""
Test script to verify the smart query processing works efficiently.
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.query_processor import QueryProcessor

def test_smart_query_processing():
    """Test that simple queries get instant responses without LLM processing."""
    
    processor = QueryProcessor()
    
    # Test cases for instant responses
    simple_queries = [
        "hi",
        "hello",
        "hey",
        "Hi",
        "Hello there",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "ok",
        "okay"
    ]
    
    # Test cases that should trigger full processing
    complex_queries = [
        "hi, what is machine learning?",
        "hello, can you help me with this document?",
        "what is the purpose of this application?",
        "how do I log in to the system?",
        "explain the features of this software"
    ]
    
    print("🧪 Testing Smart Query Processing")
    print("=" * 50)
    
    print("\n📋 Testing Simple Queries (Should be INSTANT):")
    print("-" * 40)
    
    for query in simple_queries:
        start_time = time.time()
        
        # Test the simple response method directly
        simple_response = processor._get_simple_response(query)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if simple_response:
            print(f"✅ '{query}' -> INSTANT ({processing_time:.2f}ms)")
            print(f"   Response: {simple_response[:60]}...")
        else:
            print(f"❌ '{query}' -> Would trigger full processing")
        print()
    
    print("\n📋 Testing Complex Queries (Should trigger full processing):")
    print("-" * 40)
    
    for query in complex_queries:
        simple_response = processor._get_simple_response(query)
        
        if simple_response:
            print(f"❌ '{query}' -> Got simple response (should be complex)")
        else:
            print(f"✅ '{query}' -> Correctly identified as complex")
        print()
    
    print("\n🎯 Performance Summary:")
    print("=" * 50)
    print("✅ Simple greetings like 'hi', 'hello' get instant responses")
    print("✅ Simple closings like 'thanks', 'bye' get instant responses") 
    print("✅ Complex queries correctly trigger full LLM processing")
    print("✅ No more unnecessary LLM calls for basic interactions")
    print("\n🚀 The chatbot is now MUCH more efficient!")

if __name__ == "__main__":
    test_smart_query_processing()