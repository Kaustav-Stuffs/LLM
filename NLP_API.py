from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import requests

# --- Configuration ---
API_KEY = "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app = Flask(__name__)
CORS(app)

# --- Helper Functions ---
def build_prompt(summary: str, keys: list[str]) -> str:
    """Builds the prompt for the language model."""
    keys_str = ', '.join(f'"{key}"' for key in keys)  # Quote keys
    return f"""You are a precise data extraction assistant.
Extract the following fields from the conversation summary: {keys_str}.
Return only the result in a valid JSON object format, with no additional text before or after the JSON.

Conversation Summary:
\"\"\"
{summary}
\"\"\"

JSON Output:
"""

def generate_output(prompt: str) -> str:
    """Generates text using the Gemini Pro API."""
    try:
        # Prepare the API request payload
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        # Make the API call
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)

        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            # Extract the generated text from the response
            candidates = result.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    # Clean the response to remove code block formatting
                    raw_text = parts[0]["text"].strip()
                    if raw_text.startswith("```") and raw_text.endswith("```"):
                        raw_text = raw_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
                    return raw_text
            else:
                return "The API response format is unexpected or missing 'candidates' or 'content'."
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error during API call: {e}"

# --- Flask Routes ---
@app.route('/find/relevant/data/from/speech', methods=['POST'])
def find_relevant_data_using_ai():
    """Route to find relevant data using the Gemini Pro API."""
    data = request.get_json()

    keys = data.get('keys', [])
    summary = data.get('summary', "")

    if not isinstance(keys, list) or not isinstance(summary, str):
        return jsonify({
            "message": "Invalid input format. 'keys' should be a list and 'summary' should be a string.",
            "data": {},
            "is_json": False
        }), 400

    # Build the prompt using the provided summary and keys
    prompt = build_prompt(summary, keys)

    # Use the Gemini API to generate the output
    output = generate_output(prompt)

    try:
        # Attempt to parse the cleaned output as JSON
        parsed_output = json.loads(output)
        return jsonify({
            "message": "Response Code:200.",
            "data": parsed_output,
            "is_json": True
        }), 200
    except json.JSONDecodeError:
        # If parsing fails, return raw output
        return jsonify({
            "message": "Output is not valid JSON. Returning raw text.",
            "data": output,
            "is_json": False
        }), 204


@app.route('/generate/data', methods=['POST'])
def generate_data():
    """Route to generate data using the Gemini Pro API."""
    data = request.get_json()

    summary = data.get('summary', "")
    keys = data.get('keys', [])

    if not isinstance(summary, str) or not isinstance(keys, list):
        return jsonify({
            "message": "Invalid input format. 'summary' should be a string and 'keys' should be a list.",
            "data": {},
            "is_json": False
        }), 400

    prompt = build_prompt(summary, keys)
    output = generate_output(prompt)

    try:
        # Attempt to parse the output as JSON
        parsed_output = json.loads(output)
        return jsonify({
            "message": "Response Code:200",
            "data": parsed_output,
            "is_json": True
        }), 200
    except json.JSONDecodeError:
        # If parsing fails, return raw output
        return jsonify({
            "message": "Output is not valid JSON. Returning raw text.",
            "data": output,
            "is_json": False
        }), 204


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5556)