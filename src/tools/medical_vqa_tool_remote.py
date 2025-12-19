"""
Medical Visual Question Answering Tool (Remote API)

This tool uses the HuggingFace Spaces Gradio API for medical VQA.
"""

import os
import requests
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class MedicalVQARemoteInput(BaseModel):
    """Input schema for Remote Medical VQA tool."""
    
    image_input: str = Field(
        description="File path (e.g., 'image.png') OR the magic string 'UPLOADED_IMAGE' if user uploaded a file."
    )
    question: str = Field(
        description="Question to ask about the medical image"
    )


# Remote API Configuration
REMOTE_BASE_URL = "https://simon9292-medicalqa.hf.space"
REMOTE_UPLOAD_URL = f"{REMOTE_BASE_URL}/gradio_api/upload"
REMOTE_API_URL = f"{REMOTE_BASE_URL}/gradio_api/call/gradio_vqa"


@tool(args_schema=MedicalVQARemoteInput)
def medical_vqa_remote_tool(
    image_input: str,
    question: str,
) -> str:
    """
    Answer questions about medical images using a remote multimodal AI model.
    
    This tool uploads the image to a remote VQA service and returns the analysis.
    It analyzes medical images (pathology slides, X-rays, etc.) and answers questions.
    
    IMPORTANT: The image_input can be:
    1. A file path like "image.png" for existing files
    2. "UPLOADED_IMAGE" for user-uploaded images (middleware will handle)
    
    Args:
        image_input: Either a file path OR "UPLOADED_IMAGE"
        question: Question about the medical image
    
    Returns:
        Answer to the question based on the medical image analysis
    """
    try:
        print(f"[medical_vqa_remote] Starting remote API analysis...")
        print(f"[medical_vqa_remote] Image input: {image_input}")
        print(f"[medical_vqa_remote] Question: {question}")
        
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Resolve the image path
        if not os.path.isabs(image_input):
            actual_image_path = os.path.join(project_root, image_input)
        else:
            actual_image_path = image_input
        
        # Validate file existence
        if not os.path.exists(actual_image_path):
            available = [f for f in os.listdir(project_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
            return f"Error: Image file not found at {actual_image_path}. Available images: {available}"
        
        print(f"[medical_vqa_remote] Uploading image: {actual_image_path}")
        
        # 1. Upload file to remote server
        with open(actual_image_path, "rb") as f:
            files = {'files': (os.path.basename(actual_image_path), f, 'image/png')}
            upload_response = requests.post(REMOTE_UPLOAD_URL, files=files, timeout=30)
            upload_response.raise_for_status()
            upload_result = upload_response.json()
        
        if not upload_result:
            return "Error: No file path returned from remote upload"
        
        remote_file_path = upload_result[0]
        full_file_url = f"{REMOTE_BASE_URL}/gradio_api/file={remote_file_path}"
        print(f"[medical_vqa_remote] File uploaded: {full_file_url}")
        
        # 2. Send VQA request
        payload = {
            "data": [
                {"path": full_file_url, "meta": {"_type": "gradio.FileData"}},
                question,
                128,   # max_new_tokens
                0.2,   # temperature
                0.9    # top_p
            ]
        }
        
        print(f"[medical_vqa_remote] Sending VQA request...")
        api_response = requests.post(REMOTE_API_URL, json=payload, timeout=30)
        api_response.raise_for_status()
        event_id = api_response.json().get("event_id")
        
        if not event_id:
            return "Error: No event_id returned from API"
        
        print(f"[medical_vqa_remote] Event ID: {event_id}")
        
        # 3. Poll for result
        result_url = f"{REMOTE_API_URL}/{event_id}"
        result_response = requests.get(result_url, timeout=60)
        
        # Parse SSE response
        answer = ""
        for line in result_response.text.split('\n'):
            if line.startswith('data:'):
                try:
                    data = json.loads(line[5:].strip())
                    if isinstance(data, list) and len(data) > 0:
                        answer = data[0]
                except:
                    pass
        
        if not answer:
            answer = "Unable to parse response from VQA API"
        
        print(f"[medical_vqa_remote] ✅ Got answer: {answer[:100]}..." if len(answer) > 100 else f"[medical_vqa_remote] ✅ Got answer: {answer}")
        
        return answer
        
    except requests.exceptions.Timeout:
        return "Error: Remote API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to remote VQA API: {str(e)}"
    except Exception as e:
        return f"Error during medical image analysis: {str(e)}"


if __name__ == "__main__":
    # Test the tool
    print("=== Testing Medical VQA Tool (Remote API) ===\n")
    
    test_image = "test.jpg"
    test_question = "What is shown in this medical image?"
    
    print(f"Image Input: {test_image}")
    print(f"Question: {test_question}")
    print("\nAnswer:")
    
    result = medical_vqa_remote_tool.invoke({
        "image_input": test_image,
        "question": test_question,
    })
    print(result)
