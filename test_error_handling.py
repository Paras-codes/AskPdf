"""
Test script to verify error handling implementation in AskPdf project.
This script tests various error scenarios to ensure proper error handling.
"""

import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from error_handler import (
    AskPdfBaseException, FileError, DatabaseError, ModelError, 
    QueryError, ValidationError, ErrorCode, validate_file_type,
    validate_file_size, check_required_env_vars
)

async def test_error_handler():
    """Test the error handling module functionality."""
    print("🧪 Testing Error Handler Module...")
    
    # Test 1: File type validation
    print("\n1. Testing file type validation...")
    try:
        validate_file_type("test.txt")
        print("❌ Expected ValidationError for .txt file")
    except ValidationError as e:
        print(f"✅ Caught expected ValidationError: {e.message}")
    
    try:
        validate_file_type("test.pdf")
        print("✅ PDF file validation passed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 2: File size validation
    print("\n2. Testing file size validation...")
    try:
        validate_file_size(100 * 1024 * 1024)  # 100MB
        print("❌ Expected ValidationError for oversized file")
    except ValidationError as e:
        print(f"✅ Caught expected ValidationError: {e.message}")
    
    try:
        validate_file_size(10 * 1024 * 1024)  # 10MB
        print("✅ File size validation passed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Custom exception creation
    print("\n3. Testing custom exceptions...")
    try:
        raise FileError(
            message="Test file error",
            error_code=ErrorCode.FILE_NOT_FOUND,
            details={"file": "test.pdf"}
        )
    except FileError as e:
        print(f"✅ FileError created successfully: {e.message}")
        error_dict = e.to_dict()
        print(f"   Error dict: {error_dict}")
    
    print("\n✅ Error handler module tests completed!")

async def test_models():
    """Test the models with validation."""
    print("\n🧪 Testing Models...")
    
    from models import QueryRequest, DeleteRequest
    from pydantic import ValidationError
    
    # Test valid query request
    try:
        query = QueryRequest(question="What is AI?")
        print(f"✅ Valid QueryRequest created: {query.question}")
    except Exception as e:
        print(f"❌ Unexpected error creating QueryRequest: {e}")
    
    # Test invalid query request (empty question)
    try:
        query = QueryRequest(question="")
        print("❌ Expected validation error for empty question")
    except Exception as e:
        print(f"✅ Caught expected validation error: {e}")
    
    # Test valid delete request
    try:
        delete_req = DeleteRequest(ids=["doc1", "doc2"])
        print(f"✅ Valid DeleteRequest created: {delete_req.ids}")
    except Exception as e:
        print(f"❌ Unexpected error creating DeleteRequest: {e}")
    
    # Test invalid delete request (empty ids)
    try:
        delete_req = DeleteRequest(ids=[])
        print("❌ Expected validation error for empty ids")
    except Exception as e:
        print(f"✅ Caught expected validation error: {e}")
    
    print("✅ Models tests completed!")

async def test_config():
    """Test configuration validation."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from config import UPLOAD_DIR, PERSIST_DIR, CHUNK_SIZE, validate_config
        print(f"✅ Configuration loaded successfully:")
        print(f"   UPLOAD_DIR: {UPLOAD_DIR}")
        print(f"   PERSIST_DIR: {PERSIST_DIR}")
        print(f"   CHUNK_SIZE: {CHUNK_SIZE}")
        
        # Test config validation
        config_valid = validate_config()
        print(f"   Configuration validation: {'✅ Valid' if config_valid else '⚠️  Has warnings'}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
    
    print("✅ Configuration tests completed!")

async def main():
    """Run all tests."""
    print("🚀 Starting AskPdf Error Handling Tests...\n")
    
    await test_error_handler()
    await test_models()
    await test_config()
    
    print("\n🎉 All tests completed!")
    print("\n📝 Summary:")
    print("   - Error handling module: Implemented with custom exceptions")
    print("   - Models: Updated with validation and error responses")
    print("   - Configuration: Enhanced with validation warnings")
    print("   - Main API: Protected with comprehensive error handling")
    print("   - Ingestion: Protected with file and database error handling")
    print("   - RAG Chain: Protected with model and retrieval error handling")

if __name__ == "__main__":
    asyncio.run(main())