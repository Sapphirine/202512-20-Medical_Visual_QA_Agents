import requests
import json
import os

IMAGE_PATH = "test.jpg"
BASE_URL = "https://simon9292-medicalqa.hf.space"
UPLOAD_URL = f"{BASE_URL}/gradio_api/upload"
API_URL_CALL = f"{BASE_URL}/gradio_api/call/gradio_vqa"

def run():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File {IMAGE_PATH} not found.")
        return

    # 1. Upload File
    print(f"Uploading {IMAGE_PATH} to {UPLOAD_URL}...")
    with open(IMAGE_PATH, "rb") as f:
        files = {'files': (IMAGE_PATH, f, 'image/png')}
        try:
            response = requests.post(UPLOAD_URL, files=files)
            response.raise_for_status()
            upload_resp = response.json()
            print("Upload Response:", upload_resp)
        except Exception as e:
            print(f"Upload failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(e.response.text)
            return

    # Gradi upload returns a list of paths, e.g. ['/tmp/gradio/xxx/image.png']
    if not upload_resp:
        print("No file path returned from upload")
        return
    
    remote_file_path = upload_resp[0]
    # Construct the downloadable URL or internal path
    # Often for 'path' in gradio.FileData, we can use the full URL to the file
    # The file is served at BASE_URL + /file= + remote_file_path
    # Try constructing the full URL
    
    # Note: explicit /gradio_api/ prefix might be needed or not depending on the version/proxy
    # usage in previous user command: https://simon9292-medicalqa.hf.space/gradio_api/file=/tmp/...
    # Let's try that.
    
    # Check if remote_file_path runs starts with /tmp
    full_file_url = f"{BASE_URL}/gradio_api/file={remote_file_path}"
    print(f"Using file URL: {full_file_url}")

    # 2. Prepare payload
    payload = {
        "data": [
            {"path": full_file_url, "meta": {"_type": "gradio.FileData"}},
            "does pbf show branching papillae having flbrovascular stalk covered by a single layer of cuboidal cellshaving ground-glass nuclei?",
            128,
            0.2,
            0.9
        ]
    }
    
    # 3. Send POST request
    print("Sending POST request to API...")
    try:
        response = requests.post(API_URL_CALL, json=payload)
        response.raise_for_status()
        resp_json = response.json()
        event_id = resp_json.get("event_id")
        print("Event ID:", event_id)
    except Exception as e:
        print(f"API Call failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(e.response.text)
        return

    if not event_id:
        print("No event_id returned")
        return

    # 4. Get Result
    result_url = f"{API_URL_CALL}/{event_id}"
    print(f"Polling result from: {result_url}")
    
    try:
        # Stream the response
        with requests.get(result_url, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(decoded_line)
    except Exception as e:
        print(f"Polling failed: {e}")

if __name__ == "__main__":
    run()
