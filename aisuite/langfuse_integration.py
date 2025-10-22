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

        # No verbosity controls; always log a fixed minimal subset

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
                input=self._build_input_payload(model, messages, kwargs),
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
                # Attach timing metadata as well
                trace.update(metadata={"execution_time_seconds": execution_time})
                trace.end()
                return

            # Extract response content and debug metadata
            output = self._extract_response_content(response)
            debug_meta = self._extract_debug_metadata(response)

            trace.update(
                output=output,
                level="DEFAULT",
                status_message="Success"
            )

            # Add execution time and debug metadata
            trace.update(
                metadata={
                    "execution_time_seconds": execution_time,
                    **({"debug": debug_meta} if debug_meta else {})
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
                result: Dict[str, Any] = {}

                # Message content and role
                if hasattr(choice, 'message'):
                    message = choice.message
                    result["message"] = self._serialize_message(message)

                # Finish reason and usage
                result["finish_reason"] = getattr(choice, 'finish_reason', '')
                result["usage"] = getattr(response, 'usage', {})

                return result

            return {"raw_response": str(response)}
        except Exception:
            return {"raw_response": str(response)}

    def _serialize_message(self, message: Any) -> Dict[str, Any]:
        """Serialize a message-like object to a minimal dictionary."""
        try:
            if hasattr(message, 'dict'):
                return message.dict()
            if isinstance(message, dict):
                return {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                }
            return {
                "role": getattr(message, 'role', None),
                "content": getattr(message, 'content', None),
            }
        except Exception:
            return {"raw": str(message)}

    def _extract_debug_metadata(self, response: Any) -> Optional[Dict[str, Any]]:
        """Build a compact debug structure similar to the API response 'debug' section.

        Includes tool_calls and intermediate_messages when available.
        """
        try:
            if not hasattr(response, 'choices') or not response.choices:
                return None

            choice = response.choices[0]
            debug: Dict[str, Any] = {}

            # Tool calls from the final assistant message
            if hasattr(choice, 'message'):
                message = choice.message
                tool_calls = getattr(message, 'tool_calls', None)
                if tool_calls:
                    try:
                        debug["tool_calls"] = [
                            tc.dict() if hasattr(tc, 'dict') else tc for tc in tool_calls
                        ]
                    except Exception:
                        debug["tool_calls"] = str(tool_calls)

            # Intermediate messages captured by tool runner
            intermediate_messages = getattr(choice, 'intermediate_messages', None)
            if intermediate_messages:
                try:
                    debug["intermediate_messages"] = [
                        self._serialize_message(m) for m in intermediate_messages
                    ]
                except Exception:
                    debug["intermediate_messages"] = str(intermediate_messages)

            return debug if debug else None
        except Exception:
            return None

    def _build_input_payload(self, model: str, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Build minimal input payload (include last few messages)."""
        try:
            # Only include role and last 1-2 message contents to keep payload small
            abbreviated_messages = []
            # Keep at most last 4 messages for context
            for msg in messages[-4:]:
                abbreviated_messages.append({
                    "role": msg.get("role"),
                    "content": msg.get("content") if isinstance(msg.get("content"), str) else str(msg.get("content"))
                })
            return {
                "model": model,
                "messages": abbreviated_messages,
                "parameters": params,
            }
        except Exception:
            return {"model": model, "parameters": params}

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
