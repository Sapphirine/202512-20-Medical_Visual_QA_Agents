"""
Image Handler Middleware

This middleware intercepts uploaded images in messages and saves them
to temporary files, making them accessible to tools.
"""

import os
import base64
import tempfile
import uuid
from typing import Any, Dict, Optional
from langchain.agents.middleware import AgentMiddleware


class ImageHandlerMiddleware(AgentMiddleware):
    """
    Middleware that handles image attachments in messages.
    
    When a user uploads an image, this middleware:
    1. Extracts the base64 image data from the message
    2. Saves it to a temporary file
    3. Adds the file path to the agent's state for tool access
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the middleware.
        
        Args:
            save_dir: Directory to save uploaded images. If None, uses temp directory.
        """
        self.save_dir = save_dir or tempfile.gettempdir()
        self.current_image_path: Optional[str] = None
        
    def before_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process messages before agent execution.
        
        Extracts images from messages and saves them to files.
        """
        messages = state.get("messages", [])
        
        for message in messages:
            # Check if message has image content
            content = getattr(message, "content", None)
            
            if content is None:
                continue
                
            # Handle list content (multimodal messages)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # Check for image_url format
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                            
                            if url.startswith("data:image"):
                                self._save_base64_image(url)
                                
                        # Check for direct base64 in content
                        elif item.get("type") == "image":
                            data = item.get("data", "")
                            if data:
                                self._save_base64_image(f"data:image/png;base64,{data}")
                                
            # Handle string content that might contain base64
            elif isinstance(content, str) and content.startswith("data:image"):
                self._save_base64_image(content)
                
        # Add image path to state if available
        if self.current_image_path:
            state["_uploaded_image_path"] = self.current_image_path
            
        return state
    
    def _save_base64_image(self, data_url: str) -> Optional[str]:
        """
        Save a base64-encoded image to a file.
        
        Args:
            data_url: Data URL in format "data:image/xxx;base64,..."
            
        Returns:
            Path to the saved file, or None if failed
        """
        try:
            # Parse the data URL
            if "," in data_url:
                header, data = data_url.split(",", 1)
                
                # Determine file extension from header
                if "png" in header:
                    ext = ".png"
                elif "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                elif "gif" in header:
                    ext = ".gif"
                elif "webp" in header:
                    ext = ".webp"
                else:
                    ext = ".png"  # default
            else:
                # Raw base64 data
                data = data_url
                ext = ".png"
                
            # Decode and save
            image_data = base64.b64decode(data)
            
            # Generate unique filename
            filename = f"uploaded_{uuid.uuid4().hex[:8]}{ext}"
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_data)
                
            self.current_image_path = filepath
            print(f"[ImageHandlerMiddleware] Saved uploaded image to: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"[ImageHandlerMiddleware] Error saving image: {e}")
            return None
            
    def after_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleanup after agent execution.
        
        Optionally removes temporary files.
        """
        # Reset current image path for next request
        self.current_image_path = None
        
        # Note: We don't delete the file here as it might be needed for debugging
        # In production, you might want to implement cleanup logic
        
        return state
    
    def wrap_tool_call(self, tool_call, state: Dict[str, Any]):
        """
        Intercept tool calls to inject image path if available (Sync version).
        """
        return self._inject_image_path(tool_call)

    async def awrap_tool_call(self, tool_call, state: Dict[str, Any]):
        """
        Intercept tool calls to inject image path if available (Async version).
        """
        return self._inject_image_path(tool_call)
        
    def _inject_image_path(self, tool_call):
        """
        Shared logic to inject image path into tool call.
        Handles both dictionary and object tool calls.
        """
        # If no image uploaded, return as is
        if not self.current_image_path:
            return tool_call
            
        # Handle dictionary style (standard LangChain ToolCall)
        if isinstance(tool_call, dict):
            args = tool_call.get("args", {})
            if self._update_args(args):
                tool_call["args"] = args
            return tool_call
            
        # Handle object style (e.g. ToolCallRequest)
        if hasattr(tool_call, "args"):
            args = tool_call.args
            if self._update_args(args):
                tool_call.args = args
            return tool_call
            
        return tool_call

    def _update_args(self, args: Dict[str, Any]) -> bool:
        """
        Update arguments with image path if needed.
        Returns True if changes were made.
        """
        if "image_input" in args:
            current_value = args["image_input"]
            
            import re
            
            # Pattern for uploaded files: uploaded_ + 8 hex chars + extension
            is_uploaded_pattern = isinstance(current_value, str) and bool(re.search(r"uploaded_[a-f0-9]{8}\.(png|jpg|jpeg|gif|webp)", current_value))
            
            # If it's the magic placeholder OR a made-up path OR an old uploaded file pattern
            if current_value == "UPLOADED_IMAGE" or (
                isinstance(current_value, str) and
                not os.path.exists(current_value) and 
                not current_value.startswith("data:")
            ) or (
                is_uploaded_pattern and 
                current_value != self.current_image_path
            ):
                print(f"[ImageHandlerMiddleware] Replacing '{current_value}' with '{self.current_image_path}'")
                args["image_input"] = self.current_image_path
                return True
        return False


# Convenience function to create the middleware
def create_image_handler_middleware(save_dir: Optional[str] = None) -> ImageHandlerMiddleware:
    """
    Create an image handler middleware instance.
    
    Args:
        save_dir: Directory to save uploaded images
        
    Returns:
        Configured ImageHandlerMiddleware instance
    """
    return ImageHandlerMiddleware(save_dir=save_dir)

