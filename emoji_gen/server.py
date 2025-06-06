from flask import Flask, request, jsonify
import torch
from emoji_gen.models.model_manager import model_manager
from emoji_gen.generation import generate_emoji as generate_emoji_logic
from pathlib import Path
import signal
import sys

app = Flask(__name__)

# Global server state
SERVER_STATE = {
    "model": None,
    "model_name": None
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model": SERVER_STATE["model_name"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/set-model', methods=['POST'])
def set_model():
    data = request.json
    if not data or 'model_name' not in data:
        return jsonify({"status": "error", "error": "Missing model_name parameter"}), 400
    
    model_name = data['model_name']
    try:
        # forcing cleanup of old model first to free memory
        model_manager.cleanup()

        success, message = model_manager.initialize_model(model_name)
        if success:
            model = model_manager.get_active_model()
            if model is None:
                return jsonify({"status": "error", "error": "Failed to load model"}), 500
            
            SERVER_STATE["model"] = model
            SERVER_STATE["model_name"] = model_name
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "error": message}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_emoji():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"status": "error", "error": "Missing prompt parameter"}), 400
    
    prompt = data['prompt']
    num_inference_steps = data.get('num_inference_steps', 25)
    guidance_scale = data.get('guidance_scale', 7.5)
    # num_images = data.get('num_images', 1) # functionality to save multiple images is not implemented
    output_path = data.get('output_path', None)
    use_rag = data.get('use_rag', False)
    use_llm = data.get('use_llm', False)

    if not SERVER_STATE["model"]:
        return jsonify({"status": "error", "error": "No model loaded"}), 500
    
    try:
        result = generate_emoji_logic(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_path=output_path,
            use_rag=use_rag,
            use_llm=use_llm
        )
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def start_server(model_id, host='127.0.0.1', port=5000, debug=False):
    """Start the emoji generation server with a pre-loaded model."""
    print(f"Initializing model {model_id}...")
    success, message = model_manager.initialize_model(model_id)
    if not success:
        print(f"Error: {message}")
        return
    
    SERVER_STATE["model"] = model_manager.get_active_model()
    SERVER_STATE["model_name"] = model_id
    
    print(f"Model {model_id} loaded successfully.")
    print(f"Starting server on {host}:{port}...")
    print("Press Ctrl+C to stop the server")
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        model_manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Emoji Generation Server")
    parser.add_argument("--model", type=str, required=True, help="Model ID to use for generation")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    start_server(args.model, args.host, args.port, args.debug) 