import requests
import os
from dotenv import load_dotenv

load_dotenv()
DEFAULT_SERVER_URL = os.getenv('EMOJI_SERVER_URL', 'http://127.0.0.1:5000')

def is_server_running(server_url=None):
    """Check if the emoji generation server is running."""
    if server_url is None:
        server_url = DEFAULT_SERVER_URL
    
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.ConnectionError:
        return False, None

def set_model_remote(model_name, server_url=None):
    """Set the active model on the server."""
    if server_url is None:
        server_url = DEFAULT_SERVER_URL
    
    try:
        response = requests.post(
            f"{server_url}/set-model",
            json={"model_name": model_name}
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Could not connect to server"}

def generate_emoji_remote(prompt, num_inference_steps=25, guidance_scale=7.5, num_images=1, output_path=None, server_url=None):
    """Generate an emoji using the remote server."""
    if server_url is None:
        server_url = DEFAULT_SERVER_URL
    
    try:
        response = requests.post(
            f"{server_url}/generate",
            json={
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "output_path": output_path
            }
        )
        
        if response.status_code != 200:
            return {"status": "error", "error": f"Server returned status code {response.status_code}"}
        
        try:
            data = response.json()
            if data is None:
                return {"status": "error", "error": "Server returned None response"}
            return data
        except ValueError:
            return {"status": "error", "error": "Server returned invalid JSON response"}
            
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Could not connect to server"} 