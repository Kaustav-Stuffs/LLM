import requests
import json
import os
import re



FILE_PATH= "./doc1.json"
# Function to call Gemini-2.0-Flash API
def call_gemini_api(paragraph, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Enhanced prompt for smart category and keyword selection
    prompt = f"""
    You are an expert at processing text and generating structured JSON output for the SFA (Salesforce Automation) application FAQs. Given the following paragraph, extract and format the information to match the JSON schema below. Your task is to intelligently select the `category` and `keywords` based on the paragraph's content, using context from existing FAQs.

    **JSON Schema Example**:
    ```json
    {{
        "id": "1",
        "text": "Question: What is the purpose of the SFA application? Answer: The SFA application helps track client visits, schedule meetings, record expenses, and manage salesforce tasks digitally.",
        "metadata": {{
            "source": "sfa_faq",
            "category": "Overview",
            "keywords": ["SFA", "purpose", "salesforce", "track", "meetings", "expenses", "client visits"]
        }}
    }}
    ```

    **Valid Categories** (choose one based on content):
    - Overview: General app purpose or features (e.g., "What is the SFA app?").
    - Login: Authentication or login processes (e.g., "How do I log in?").
    - Meetings: Creating, scheduling, or managing meetings (e.g., "How to create a meeting?").
    - Outdoor Duty: Outdoor activities, GPS tracking, routes (e.g., "How to start Outdoor Duty?").
    - Expenses: Expense logging or approval (e.g., "How to add expenses?").
    - Clients/Leads: Managing clients or leads (e.g., "How to create a new client?").
    - Follow-Ups: Follow-up meeting actions (e.g., "How to reschedule a meeting?").
    - Attendance: Attendance tracking (e.g., "How to mark attendance?").

    **Input Paragraph**:
    {paragraph}

    **Instructions**:
    1. **Extract Question and Answer**:
       - Identify the question and answer. If not explicitly labeled, infer them from the content (e.g., first sentence as question, rest as answer).
       - Format the `text` field as "Question: <question> Answer: <answer>".
       - Ensure the question is concise and the answer is clear, matching the style of the example.

    2. **Select Category**:
       - Analyze the paragraph's content to choose the most relevant category from the list above.
       - Use key indicators:
         - GPS, routes, Toggle Button, OD → Outdoor Duty
         - Meetings, follow-ups, scheduling → Meetings or Follow-Ups
         - Expenses, approvals → Expenses
         - Clients, leads, branches → Clients/Leads
         - Login, credentials, Company Identifier → Login
         - Attendance, HRMS → Attendance
         - General app description → Overview
       - If ambiguous, default to "Overview".
       - Example: A paragraph about GPS tracking for client visits should select "Outdoor Duty".

    3. **Generate Keywords**:
       - Extract 3–10 keywords that capture the paragraph's core concepts.
       - Prioritize specific terms from the SFA app (e.g., "Outdoor Duty", "Toggle Button", "Company Identifier") and avoid generic words (e.g., "app", "use").
       - Ensure keywords are consistent with existing FAQs (e.g., use "meeting" or "meetings" consistently, prefer "Outdoor Duty" over "OD").
       - Example: For a paragraph about starting Outdoor Duty, keywords might include ["Outdoor Duty", "Toggle Button", "GPS", "Dashboard"].
       - Match the style of the example keywords (lowercase, descriptive).

    4. **Set Source**:
       - Always set `source` to "sfa_faq".

    5. **Output**:
       - Return only the JSON object with `text` and `metadata` fields, formatted as a valid JSON string.
       - Do NOT generate the `id` field; it will be handled separately.
       - Ensure the JSON is valid and matches the schema.

    **Example Input and Output**:
    - **Input**: "How can I start my Outdoor Duty in the SFA app? To start, tap the Toggle Button on the Dashboard. GPS will track your route."
    - **Output**:
      ```json
      {{
        "text": "Question: How can I start my Outdoor Duty in the SFA app? Answer: To start, tap the Toggle Button on the Dashboard. GPS will track your route.",
        "metadata": {{
          "source": "sfa_faq",
          "category": "Outdoor Duty",
          "keywords": ["Outdoor Duty", "Toggle Button", "GPS", "Dashboard", "route"]
        }}
      }}
      ```

    **Output**:
    Return a JSON string matching the schema (without the `id` field).
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        # Extract the generated content from the response
        generated_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        return generated_text
    except requests.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "{}"

# Function to convert paragraph and append to doc1.json
def convert_paragraph_to_json(paragraph, json_file_path=FILE_PATH):
    # Get API key from environment variable
    api_key = "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA"
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Read the existing doc1.json file
    try:
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        if not isinstance(json_data, list):
            raise ValueError("doc1.json does not contain a JSON array")
    except FileNotFoundError:
        print(f"Warning: {json_file_path} not found. Creating new file.")
        json_data = []
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} contains invalid JSON. Starting with empty array.")
        json_data = []
    
    # Determine the next ID
    if json_data:
        last_id = max(int(item["id"]) for item in json_data)
        new_id = str(last_id + 1)
    else:
        new_id = "1"
    
    # Call Gemini API to process the paragraph
    gemini_response = call_gemini_api(paragraph, api_key)

    def extract_json_from_response(response_text):
        # Try to extract JSON from code block or plain text
        # 1. Look for triple backtick code block with json
        code_block_match = re.search(r"```(?:json)?\s*({.*?})\s*```", response_text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
        # 2. Look for first JSON object in the text
        brace_match = re.search(r"({.*})", response_text, re.DOTALL)
        if brace_match:
            return brace_match.group(1)
        # 3. Fallback: return as is
        return response_text

    try:
        # Extract JSON string from Gemini response
        json_str = extract_json_from_response(gemini_response)
        parsed_response = json.loads(json_str)
    except Exception:
        print("Error: Gemini API did not return valid JSON")
        parsed_response = {
            "text": f"Question: Unknown Answer: Unable to process paragraph",
            "metadata": {
                "source": "sfa_faq",
                "category": "Overview",
                "keywords": ["error", "unknown"]
            }
        }
    
    # Construct the new JSON object
    new_entry = {
        "id": new_id,
        "text": parsed_response.get("text", "Question: Unknown Answer: Unable to process paragraph"),
        "metadata": parsed_response.get("metadata", {
            "source": "sfa_faq",
            "category": "Overview",
            "keywords": ["error", "unknown"]
        })
    }
    
    # Append the new entry to the JSON data
    json_data.append(new_entry)
    
    # Write the updated JSON back to the file
    try:
        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Successfully appended new entry to {json_file_path}")
    except Exception as e:
        print(f"Error writing to {json_file_path}: {e}")
    
    return new_entry

# Example usage
if __name__ == "__main__":
    # Sample paragraph
    sample_paragraph = """
        SFA stands for Salesforce Automation. It is a digital tool that helps sales teams manage their activities, including tracking client visits, scheduling meetings, recording expenses, and managing tasks. The SFA application is designed to streamline the sales process and improve efficiency.
        To log in to the SFA application, users need to enter their credentials and the Company Identifier. Once logged in, they can access various features such as creating new clients, managing meetings, and logging expenses.
        
          """
    
    try:
        # Convert paragraph and append to doc1.json
        result = convert_paragraph_to_json(sample_paragraph)
        
        # Print the new entry (for debugging)
        print("New entry added:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error processing paragraph: {e}")