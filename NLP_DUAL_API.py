from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
import requests
import time

# --- Configuration ---
PRIMARY_API_KEY = "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA"
BACKUP_API_KEY = "AIzaSyBackupKey1234567890"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app = Flask(__name__)
CORS(app)

# Initialize Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["15 per minute", "1500 per day"]  # Enforce RPM and RPD limits
)

# Token and request usage tracking
TOKEN_LIMIT_PER_MINUTE = 1_000_000
REQUEST_LIMIT_PER_DAY = 1500
usage_stats = {
    "current_minute": int(time.time() // 60),
    "current_day": int(time.time() // (60 * 60 * 24)),
    "tokens_used": 0,
    "requests_made": 0
}


# --- Helper Functions ---
def track_usage(tokens: int) -> bool:
    """
    Tracks token and request usage and ensures limits are not exceeded.
    Returns True if usage is within limits, False otherwise.
    """
    global usage_stats
    current_minute = int(time.time() // 60)
    current_day = int(time.time() // (60 * 60 * 24))

    # Reset token usage if a new minute starts
    if usage_stats["current_minute"] != current_minute:
        usage_stats["current_minute"] = current_minute
        usage_stats["tokens_used"] = 0

    # Reset request usage if a new day starts
    if usage_stats["current_day"] != current_day:
        usage_stats["current_day"] = current_day
        usage_stats["requests_made"] = 0

    # Check if adding the current tokens or requests exceeds the limits
    if usage_stats["tokens_used"] + tokens > TOKEN_LIMIT_PER_MINUTE:
        return False
    if usage_stats["requests_made"] + 1 > REQUEST_LIMIT_PER_DAY:
        return False

    # Update usage stats
    usage_stats["tokens_used"] += tokens
    usage_stats["requests_made"] += 1
    return True


def calculate_token_count(prompt: str) -> int:
    """
    Calculates the number of tokens in the given prompt.
    For simplicity, assume 1 token per word.
    """
    return len(prompt.split())


def build_prompt(summary: str, keys: list[str]) -> str:
    """
    Builds the prompt for the language model.
    If keys are not provided, they will be automatically extracted.
    """
    # if not keys:
    #     keys = ["participants", "date", "time", "location", "topic", "main_points", "decisions", 
    #             "action_items", "deadlines", "responsible_parties", "follow_up", "concerns", 
    #             "agreements", "resources_mentioned", "metrics", "project_name", "status", 
    #             "budget", "timeline", "next_steps", "key_insights"]
    
    keys_str = ', '.join(f'"{key}"' for key in keys)  # Quote keys
    return f"""You are a precise data extraction assistant.
Extract the following fields from the conversation summary in the exact order provided: {keys_str}.
Whatever the Input Language is, the Output Language should be English.
If keys are not given then use your intelligence to extract the most relevant data possible.
Return only the result in a valid JSON object format, with no additional text before or after the JSON.

Conversation Summary:
\"\"\"
{summary}
\"\"\"

JSON Output:
"""


def generate_output(prompt: str, api_key: str) -> str:
    """Generates text using the specified API key."""
    try:
        # Calculate token count for the prompt
        token_count = calculate_token_count(prompt)

        # Check usage limits
        if not track_usage(token_count):
            return "Usage limits exceeded. Switching to backup API."

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
        response = requests.post(f"{API_URL}?key={api_key}", headers=headers, json=payload)

        # Handle rate limit errors
        if response.status_code == 429:
            output = generate_output(prompt, BACKUP_API_KEY)
            return "Rate limit exceeded."

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
@limiter.limit("15 per minute")  # Enforce RPM limit
def find_relevant_data_using_ai():
    """Route to find relevant data using the Gemini Pro API."""
    data = request.get_json()
    summary = data.get('summary', "")
    keys = data.get('keys', [])

    if not isinstance(keys, list) or not isinstance(summary, str):
        return jsonify({
            "message": "Invalid input format. 'keys' should be a list and 'summary' should be a string.",
            "data": {},
            "is_json": False
        }), 400

    # Build the prompt using the provided summary and keys
    prompt = build_prompt(summary, keys)

    # Use the primary API to generate the output
    global output
    output = generate_output(prompt, PRIMARY_API_KEY)

    # If usage limits are exceeded, switch to the backup API
    if "Usage limits exceeded" in output:
        output = generate_output(prompt, BACKUP_API_KEY)
    elif "Rate limit exceeded" in output:
        output = generate_output(prompt, BACKUP_API_KEY)
    elif "API Error" in output:
        output = generate_output(prompt, BACKUP_API_KEY)    
    try:
        # Attempt to parse the cleaned output as JSON
        parsed_output = json.loads(output)
        return jsonify({
            "message": "Response Code:200.",
            "data": parsed_output,
            # "usage_stats": usage_stats,  # Show usage stats for each attempt
            "is_json": True
        }), 200
    except json.JSONDecodeError:
        # If parsing fails, return raw output
        return jsonify({
            "message": "Output is not valid JSON. Returning raw text.",
            "data": output,
            # "usage_stats": usage_stats,  # Show usage stats for each attempt
            "is_json": False
        }), 204


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)