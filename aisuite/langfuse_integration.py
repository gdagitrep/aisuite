"""
Langfuse integration for aisuite logging.
This module provides enhanced observability by integrating Langfuse tracing with aisuite's logging.
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from functools import wraps

def _check_langfuse_availability():
    """Check if Langfuse is available at runtime."""
    try:
        from langfuse import Langfuse
        return True, Langfuse, None
    except ImportError:
        return False, None, None

# Check availability at import time
LANGFUSE_AVAILABLE, Langfuse, observe = _check_langfuse_availability()


class LangfuseIntegration:
    """Integration class for Langfuse observability with aisuite."""
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 public_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize Langfuse integration.
        
        Args:
            secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
            public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
            base_url: Langfuse base URL (defaults to LANGFUSE_BASE_URL env var)
            enabled: Whether to enable Langfuse tracing
        """
        # Check Langfuse availability at runtime
        langfuse_available, Langfuse, observe = _check_langfuse_availability()
        self.enabled = enabled and langfuse_available
        
        if not self.enabled:
            self.langfuse = None
            return
            
        # Get credentials from environment or parameters
        self.secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
        self.public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
        self.base_url = base_url or os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')
        
        if not self.secret_key or not self.public_key:
            print("Warning: Langfuse credentials not found. Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables.")
            self.enabled = False
            self.langfuse = None
            return
            
        try:
            # Initialize Langfuse with correct parameters
            self.langfuse = Langfuse(
                secret_key=self.secret_key,
                public_key=self.public_key,
                host=self.base_url
            )
            print("Langfuse integration initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Langfuse: {e}")
            self.enabled = False
            self.langfuse = None

    def trace_llm_call(self, 
                      method_name: str,
                      model: str,
                      messages: List[Dict[str, Any]],
                      **kwargs) -> Optional[Any]:
        """
        Trace an LLM call with Langfuse.
        
        Args:
            method_name: Name of the method being called
            model: Model being used
            messages: Messages sent to the LLM
            **kwargs: Additional parameters
            
        Returns:
            Langfuse trace object if enabled, None otherwise
        """
        if not self.enabled or not self.langfuse:
            return None
            
        try:
            # Create a span for this LLM call
            span = self.langfuse.start_span(
                name=f"aisuite_{method_name}",
                input={
                    "model": model,
                    "messages": messages,
                    "parameters": kwargs
                },
                metadata={
                    "provider": "aisuite",
                    "method": method_name,
                    "model": model
                }
            )
            
            return span
        except Exception as e:
            print(f"Failed to create Langfuse span: {e}")
            return None

    def update_trace_with_response(self, 
                                 trace: Any,
                                 response: Any,
                                 execution_time: float,
                                 error: Optional[Exception] = None):
        """
        Update the trace with the response.
        
        Args:
            trace: Langfuse span object
            response: Response from the LLM
            execution_time: Time taken for execution
            error: Any error that occurred
        """
        if not self.enabled or not trace:
            return
            
        try:
            # Update the span with output data
            if error:
                trace.update(
                    output={"error": str(error)},
                    level="ERROR",
                    status_message=str(error)
                )
            else:
                # Extract response content
                output = self._extract_response_content(response)
                trace.update(
                    output=output,
                    level="DEFAULT",
                    status_message="Success"
                )
            
            # Add execution time as metadata
            trace.update(
                metadata={
                    "execution_time_seconds": execution_time
                }
            )
            
            # End the span
            trace.end()
            
        except Exception as e:
            print(f"Failed to update Langfuse span: {e}")

    def _extract_response_content(self, response: Any) -> Dict[str, Any]:
        """Extract meaningful content from the response for Langfuse."""
        try:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message'):
                    message = choice.message
                    return {
                        "content": getattr(message, 'content', ''),
                        "role": getattr(message, 'role', 'assistant'),
                        "finish_reason": getattr(choice, 'finish_reason', ''),
                        "usage": getattr(response, 'usage', {})
                    }
            return {"raw_response": str(response)}
        except Exception:
            return {"raw_response": str(response)}

    def flush(self):
        """Flush any pending traces to Langfuse."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                print(f"Failed to flush Langfuse traces: {e}")


# Global Langfuse integration instance
_langfuse_integration: Optional[LangfuseIntegration] = None


def get_langfuse_integration() -> LangfuseIntegration:
    """Get the global Langfuse integration instance."""
    global _langfuse_integration
    if _langfuse_integration is None:
        _langfuse_integration = LangfuseIntegration()
    return _langfuse_integration


def langfuse_trace_llm_call(method_name: str, model: str, messages: List[Dict[str, Any]], **kwargs):
    """
    Decorator to trace LLM calls with Langfuse.
    
    Usage:
        @langfuse_trace_llm_call("chat_completions_create", "gpt-3.5-turbo", messages)
        def my_llm_call():
            # Your LLM call here
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            integration = get_langfuse_integration()
            trace = integration.trace_llm_call(method_name, model, messages, **kwargs)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                integration.update_trace_with_response(trace, result, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                integration.update_trace_with_response(trace, None, execution_time, e)
                raise
        return wrapper
    return decorator
