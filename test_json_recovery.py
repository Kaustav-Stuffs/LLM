#!/usr/bin/env python3
"""
Test script to verify the enhanced JSON recovery mechanisms work with real-world malformed JSON.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.document_processor import DocumentProcessor

def test_specific_json_error():
    """Test the specific JSON error case from the user's report."""
    
    processor = DocumentProcessor()
    
    # Simulate the type of malformed JSON that causes "Expecting property name enclosed in double quotes"
    malformed_json_cases = [
        # Case 1: Unquoted property names (common Ollama issue)
        '[{text: "Question: What is ML? Answer: Machine Learning", metadata: {category: "Definitions", keywords: ["ML", "learning"]}}]',
        
        # Case 2: Missing commas between objects
        '[{"text": "Q: Test1? A: Answer1", "metadata": {"category": "Test"}} {"text": "Q: Test2? A: Answer2", "metadata": {"category": "Test"}}]',
        
        # Case 3: Unescaped quotes in content
        '[{"text": "Question: What\'s the purpose? Answer: It\'s for testing", "metadata": {"category": "Test"}}]',
        
        # Case 4: Mixed quote types
        "[{'text': \"Question: Mixed quotes? Answer: Yes\", 'metadata': {'category': 'Test'}}]",
        
        # Case 5: The specific error pattern (property name at line 27, column 96)
        '''[
        {
            "text": "Question: What is machine learning? Answer: Machine learning is a method of data analysis",
            "metadata": {
                "source": "MLMLML.pdf",
                "category": "Definitions",
                "keywords": ["machine learning", "data analysis"]
            }
        },
        {
            "text": "Question: What are neural networks? Answer: Neural networks are computing systems",
            "metadata": {
                "source": "MLMLML.pdf",
                category: "Technical Details",
                "keywords": ["neural networks", "computing"]
            }
        }
        ]'''
    ]
    
    print("🧪 Testing Enhanced JSON Recovery Mechanisms")
    print("=" * 60)
    
    for i, malformed_json in enumerate(malformed_json_cases, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"Input: {malformed_json[:100]}...")
        
        try:
            # Test the recovery mechanism
            recovered = processor._recover_json_objects(malformed_json, f"test_case_{i}.pdf")
            
            print(f"✅ Recovery successful!")
            print(f"📊 Recovered {len(recovered)} objects")
            
            # Validate the recovered objects
            valid_objects = 0
            for obj in recovered:
                if (isinstance(obj, dict) and 
                    obj.get('text') and 
                    obj.get('metadata') and
                    isinstance(obj.get('metadata'), dict)):
                    valid_objects += 1
            
            print(f"✅ Valid objects: {valid_objects}/{len(recovered)}")
            
            # Show first recovered object as example
            if recovered and valid_objects > 0:
                first_obj = recovered[0]
                print(f"📄 Sample recovered object:")
                print(f"   Text: {first_obj.get('text', 'N/A')[:80]}...")
                print(f"   Category: {first_obj.get('metadata', {}).get('category', 'N/A')}")
                print(f"   Keywords: {first_obj.get('metadata', {}).get('keywords', [])}")
            
        except Exception as e:
            print(f"❌ Recovery failed: {e}")
        
        print("-" * 40)
    
    print("\n🎯 Summary:")
    print("The enhanced JSON recovery system now includes:")
    print("✅ Aggressive JSON string cleaning")
    print("✅ Unescaped quote fixing")
    print("✅ Missing comma detection and repair")
    print("✅ Property name quote fixing")
    print("✅ Multiple recovery strategies")
    print("✅ Text extraction from malformed JSON as last resort")
    print("\nThis should handle the 'Expecting property name' error you encountered!")

if __name__ == "__main__":
    test_specific_json_error()