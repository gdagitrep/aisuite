"""
Official Langfuse SDK integration for aisuite.

This module provides a clean integration with the official Langfuse SDK
instead of our custom implementation.
"""

import os
from typing import Optional, Any
from dotenv import load_dotenv

def get_langfuse_client():
    """
    Get the official Langfuse OpenAI client.
    
    Returns:
        OpenAI client with Langfuse tracing enabled, or None if not available
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Check if Langfuse credentials are available
        secret_key = os.getenv('LANGFUSE_SECRET_KEY')
        public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
        base_url = os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')
        
        if not secret_key or not public_key:
            print("Warning: Langfuse credentials not found. Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables.")
            return None
            
        # Import the official Langfuse OpenAI client
        from langfuse.openai import openai
        
        # Set the environment variables for Langfuse
        os.environ['LANGFUSE_SECRET_KEY'] = secret_key
        os.environ['LANGFUSE_PUBLIC_KEY'] = public_key
        os.environ['LANGFUSE_HOST'] = base_url
        
        print("Official Langfuse SDK initialized successfully")
        return openai
        
    except ImportError:
        print("Warning: Langfuse not installed. Install with: pip install langfuse")
        return None
    except Exception as e:
        print(f"Failed to initialize Langfuse: {e}")
        return None

def is_langfuse_available():
    """Check if Langfuse is available and properly configured."""
    try:
        from langfuse.openai import openai
        return True
    except ImportError:
        return False

def get_langfuse_traced_client(original_client):
    """
    Replace the original OpenAI client with Langfuse-traced version.
    
    Args:
        original_client: The original OpenAI client
        
    Returns:
        Langfuse-traced OpenAI client or original client if Langfuse not available
    """
    langfuse_client = get_langfuse_client()
    if langfuse_client:
        return langfuse_client
    return original_client
