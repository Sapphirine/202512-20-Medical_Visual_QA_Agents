"""
Medical Visual Question Answering Tool

This tool wraps the multimodal inference model to answer questions about medical images.
Supports both file paths and base64-encoded images.
"""

import os
import base64
import tempfile
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class MedicalVQAInput(BaseModel):
    """Input schema for Medical VQA tool."""
    
    image_input: str = Field(
        description="File path (e.g., 'image.png') OR the magic string 'UPLOADED_IMAGE' if user uploaded a file."
    )
    question: str = Field(
        description="Question to ask about the medical image"
    )
    projector_checkpoint: Optional[str] = Field(
        default="newlytrained/projector_epoch1.pt",
        description="Path to the trained projector checkpoint file"
    )


@tool(args_schema=MedicalVQAInput)
def medical_vqa_tool(
    image_input: str,
    question: str,
    projector_checkpoint: str = "newlytrained/projector_epoch1.pt"
) -> str:
    """
    Answer questions about medical images using a multimodal AI model.
    
    This tool combines vision and language models to analyze medical images
    (pathology slides, X-rays, etc.) and provide answers to questions about them.
    
    IMPORTANT: The image_input can be:
    1. A file path like "image.png" for existing files
    2. Base64-encoded image data (if user uploaded an image, pass the base64 data)
    
    Use this tool when you need to:
    - Identify tissue types in pathology images
    - Describe medical imaging findings
    - Answer diagnostic questions about medical images
    - Analyze organ structures in medical scans
    
    Args:
        image_input: Either a file path OR base64-encoded image data
        question: Question about the medical image
        projector_checkpoint: Path to trained projector model (default: projector_epoch2.pt)
    
    Returns:
        Answer to the question based on the medical image analysis
    
    Example with file path:
        >>> result = medical_vqa_tool(image_input="image.png", question="What tissue is shown?")
    
    Example with uploaded image:
        >>> result = medical_vqa_tool(image_input="UPLOADED_IMAGE", question="What is this?")
    """
    # Import the inference function (lazy import to avoid loading models at startup)
    try:
        print("[medical_vqa_tool] Starting image analysis...")
        
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Add project root to path if needed
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from inference import infer
        
        # Handle base64 encoded images or file paths
        actual_image_path = None
        temp_file = None
        
        # Check input type
        is_base64 = image_input.startswith("data:image") or (
            len(image_input) > 500 and "/" not in image_input and "\\" not in image_input
        )
        
        if is_base64:
            print(f"[medical_vqa_tool] Detected base64 image (length: {len(image_input)} chars)")
        
        if image_input.startswith("data:image"):
            # Extract base64 data from data URL (format: data:image/png;base64,xxxx)
            try:
                header, data = image_input.split(",", 1)
                print(f"[medical_vqa_tool] Decoding base64 data...")
                image_data = base64.b64decode(data)
                print(f"[medical_vqa_tool] Decoded {len(image_data)} bytes")
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_file.write(image_data)
                temp_file.close()
                actual_image_path = temp_file.name
                print(f"[medical_vqa_tool] Saved to temp file: {actual_image_path}")
            except Exception as e:
                return f"Error decoding base64 image data: {str(e)}"
            
        elif len(image_input) > 500 and "/" not in image_input and "\\" not in image_input and "." not in image_input[:20]:
            # Likely raw base64 data (long string without path separators)
            try:
                image_data = base64.b64decode(image_input)
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file.write(image_data)
                temp_file.close()
                actual_image_path = temp_file.name
            except:
                # Not valid base64, treat as path
                actual_image_path = image_input
        else:
            # Regular file path
            actual_image_path = image_input
        
        # Resolve paths relative to project root
        if not os.path.isabs(actual_image_path):
            actual_image_path = os.path.join(project_root, actual_image_path)
        
        if not os.path.isabs(projector_checkpoint):
            projector_checkpoint = os.path.join(project_root, projector_checkpoint)
        
        # Validate file existence
        if not os.path.exists(actual_image_path):
            # List available images
            available = [f for f in os.listdir(project_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
            return f"Error: Image file not found at {actual_image_path}. Available images: {available}"
        
        if not os.path.exists(projector_checkpoint):
            return f"Error: Projector checkpoint not found at {projector_checkpoint}"
        
        # Run inference
        print(f"[medical_vqa_tool] Running inference...")
        print(f"[medical_vqa_tool] Image: {actual_image_path}")
        print(f"[medical_vqa_tool] Question: {question}")
        
        answer = infer(
            image=actual_image_path,
            question=question,
            projector_ckpt=projector_checkpoint
        )
        
        print(f"[medical_vqa_tool] ✅ Got answer: {answer[:100]}..." if len(answer) > 100 else f"[medical_vqa_tool] ✅ Got answer: {answer}")
        
        # Clean up temp file if created
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return answer
        
    except Exception as e:
        return f"Error during medical image analysis: {str(e)}"


# For backward compatibility and direct testing
def run_medical_vqa(image_input: str, question: str, checkpoint: str = "newlytrained/projector_epoch1.pt") -> str:
    """Convenience function for direct calls."""
    return medical_vqa_tool.invoke({
        "image_input": image_input,
        "question": question,
        "projector_checkpoint": checkpoint
    })


if __name__ == "__main__":
    # Test the tool
    print("=== Testing Medical VQA Tool ===\n")
    
    test_image = "image.png"
    test_question = "What is shown in this pathology image?"
    
    print(f"Image Input: {test_image}")
    print(f"Question: {test_question}")
    print("\nAnswer:")
    
    result = run_medical_vqa(test_image, test_question)
    print(result)

