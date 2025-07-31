# import requests
# import json
# import os
# import re



# FILE_PATH= "./doc1.json"
# # Function to call Gemini-2.0-Flash API
# def call_gemini_api(paragraph, api_key):
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
#     headers = {
#         "Content-Type": "application/json"
#     }
    
#         # Enhanced prompt for splitting long summaries into multiple entries
#     prompt = f"""
#     You are an expert at processing text and generating structured JSON output for the SFA (Salesforce Automation) application FAQs. Given the following paragraph or summary, your task is to intelligently split the content into multiple FAQ-style entries if it contains information about more than one topic, question, or answer. Each entry should match the JSON schema below and focus on a single question and answer pair.

#     **JSON Schema Example**:
#     ```json
#     {{
#         "text": "Question: What is the purpose of the SFA application? Answer: The SFA application helps track client visits, schedule meetings, record expenses, and manage salesforce tasks digitally.",
#         "metadata": {{
#             "source": "sfa_faq",
#             "category": "Overview",
#             "keywords": ["SFA", "purpose", "salesforce", "track", "meetings", "expenses", "client visits"]
#         }}
#     }}
#     ```

#     **Valid Categories** (choose one based on content):
#     - Overview: General app purpose or features (e.g., "What is the SFA app?").
#     - Login: Authentication or login processes (e.g., "How do I log in?").
#     - Meetings: Creating, scheduling, or managing meetings (e.g., "How to create a meeting?").
#     - Outdoor Duty: Outdoor activities, GPS tracking, routes (e.g., "How to start Outdoor Duty?").
#     - Expenses: Expense logging or approval (e.g., "How to add expenses?").
#     - Clients/Leads: Managing clients or leads (e.g., "How to create a new client?").
#     - Follow-Ups: Follow-up meeting actions (e.g., "How to reschedule a meeting?").
#     - Attendance: Attendance tracking (e.g., "How to mark attendance?").

#     **Instructions**:
#     1. **Split the Input**: If the paragraph contains information about multiple topics, questions, or answers, split it into separate entries. Each entry should focus on a single question and answer pair.
#     2. **Extract Question and Answer**: For each entry, identify the question and answer. If not explicitly labeled, infer them from the content.
#     3. **Select Category**: Choose the most relevant category for each entry from the list above.
#     4. **Generate Keywords**: Extract 3–10 keywords for each entry that capture its core concepts.
#     5. **Set Source**: Always set `source` to "sfa_faq".
#     6. **Output**: Return a JSON array (list) of objects, each matching the schema above (without the `id` field). Do NOT merge multiple topics into one entry.

#     **Example Input**:
#     "SFA stands for Salesforce Automation. It is a digital tool that helps sales teams manage their activities, including tracking client visits, scheduling meetings, recording expenses, and managing tasks. To log in to the SFA application, users need to enter their credentials and the Company Identifier."

#     **Example Output**:
#     ```json
#     [
#       {{
#         "text": "Question: What is the purpose of the SFA application? Answer: The SFA application helps sales teams manage activities such as tracking client visits, scheduling meetings, recording expenses, and managing tasks.",
#         "metadata": {{
#           "source": "sfa_faq",
#           "category": "Overview",
#           "keywords": ["SFA", "purpose", "salesforce", "tracking", "meetings", "expenses", "tasks"]
#         }}
#       }},
#       {{
#         "text": "Question: How do I log in to the SFA application? Answer: Enter your credentials and the Company Identifier to log in.",
#         "metadata": {{
#           "source": "sfa_faq",
#           "category": "Login",
#           "keywords": ["login", "credentials", "Company Identifier", "SFA"]
#         }}
#       }}
#     ]
#     ```

#     **Input Paragraph**:
#     {paragraph}

#     **Output**:
#     Return a JSON array (list) of objects, each matching the schema above (without the `id` field).
#     """
    
#     payload = {
#         "contents": [{
#             "parts": [{"text": prompt}]
#         }]
#     }
    
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         response_data = response.json()
#         # Extract the generated content from the response
#         generated_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "[]")
#         return generated_text
#     except requests.RequestException as e:
#         print(f"Error calling Gemini API: {e}")
#         return "[]"

# # Function to convert paragraph and append to doc1.json
# def convert_paragraph_to_json(paragraph, json_file_path=FILE_PATH):
#     api_key = "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA"
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY environment variable not set")
    
#     # Read the existing doc1.json file
#     try:
#         with open(json_file_path, "r") as f:
#             json_data = json.load(f)
#         if not isinstance(json_data, list):
#             raise ValueError("doc1.json does not contain a JSON array")
#     except FileNotFoundError:
#         print(f"Warning: {json_file_path} not found. Creating new file.")
#         json_data = []
#     except json.JSONDecodeError:
#         print(f"Error: {json_file_path} contains invalid JSON. Starting with empty array.")
#         json_data = []
    
#     # Determine the next ID
#     if json_data:
#         last_id = max(int(item["id"]) for item in json_data)
#     else:
#         last_id = 0
    
#     # Call Gemini API to process the paragraph
#     gemini_response = call_gemini_api(paragraph, api_key)

#     def extract_json_from_response(response_text):
#         # Try to extract JSON array from code block or plain text
#         # 1. Look for triple backtick code block with json array
#         code_block_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response_text)
#         if code_block_match:
#             return code_block_match.group(1)
#         # 2. Look for first JSON array in the text
#         array_match = re.search(r"(\[[\s\S]*\])", response_text)
#         if array_match:
#             return array_match.group(1)
#         # 3. Fallback: return as is
#         return response_text

#     try:
#         # Extract JSON string from Gemini response
#         json_str = extract_json_from_response(gemini_response)
#         parsed_response = json.loads(json_str)
#         if isinstance(parsed_response, dict):
#             parsed_response = [parsed_response]  # Make it a list for consistency
#     except Exception:
#         print("Error: Gemini API did not return valid JSON")
#         parsed_response = [{
#             "text": f"Question: Unknown Answer: Unable to process paragraph",
#             "metadata": {
#                 "source": "sfa_faq",
#                 "category": "Overview",
#                 "keywords": ["error", "unknown"]
#             }
#         }]
    
#     # Construct new JSON objects with unique IDs
#     new_entries = []
#     for entry in parsed_response:
#         last_id += 1
#         new_entry = {
#             "id": str(last_id),
#             "text": entry.get("text", "Question: Unknown Answer: Unable to process paragraph"),
#             "metadata": entry.get("metadata", {
#                 "source": "sfa_faq",
#                 "category": "Overview",
#                 "keywords": ["error", "unknown"]
#             })
#         }
#         json_data.append(new_entry)
#         new_entries.append(new_entry)
    
#     # Write the updated JSON back to the file
#     try:
#         with open(json_file_path, "w") as f:
#             json.dump(json_data, f, indent=2)
#         print(f"Successfully appended {len(new_entries)} new entr{'y' if len(new_entries)==1 else 'ies'} to {json_file_path}")
#     except Exception as e:
#         print(f"Error writing to {json_file_path}: {e}")
    
#     return new_entries

# # Example usage
# if __name__ == "__main__":
#     # Sample paragraph
#     sample_paragraph = """
#         SFA stands for Salesforce Automation. It is a digital tool that helps sales teams manage their activities, including tracking client visits, scheduling meetings, recording expenses, and managing tasks. The SFA application is designed to streamline the sales process and improve efficiency.
#         To log in to the SFA application, users need to enter their credentials and the Company Identifier. Once logged in, they can access various features such as creating new clients, managing meetings, and logging expenses.
        
#           """
    
#     try:
#         # Convert paragraph and append to doc1.json
#         result = convert_paragraph_to_json(sample_paragraph)
        
#         # Print the new entry (for debugging)
#         print("New entry added:")
#         print(json.dumps(result, indent=2))
        
#     except Exception as e:
#         print(f"Error processing paragraph: {e}")

import requests
import json
import os
import re

FILE_PATH = "./doc1.json"

# Function to call Gemini-2.0-Flash API
def call_gemini_api(paragraph, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Enhanced prompt for splitting long summaries into multiple entries and extracting topic
    prompt = f"""
    You are an expert at processing text and generating structured JSON output for FAQs based on the provided paragraph. Your task is to:
    1. Identify the main topic of the paragraph (e.g., 'Salesforce Automation', 'Customer Support', 'Inventory Management').
    2. Split the content into multiple FAQ-style entries if it contains information about more than one topic, question, or answer. Each entry should match the JSON schema below and focus on a single question and answer pair.

    **JSON Schema Example**:
    ```json
    {{
        "topic": "Salesforce Automation",
        "entries": [
            {{
                "text": "Question: What is the purpose of the SFA application? Answer: The SFA application helps track client visits, schedule meetings, record expenses, and manage salesforce tasks digitally.",
                "metadata": {{
                    "source": "faq",
                    "category": "Overview",
                    "keywords": ["SFA", "purpose", "salesforce", "track", "meetings", "expenses", "client visits"]
                }}
            }}
        ]
    }}
    ```

    **Valid Categories** (choose one based on content):
    - Overview: General app purpose or features.
    - Login: Authentication or login processes.
    - Meetings: Creating, scheduling, or managing meetings.
    - Outdoor Duty: Outdoor activities, GPS tracking, routes.
    - Expenses: Expense logging or approval.
    - Clients/Leads: Managing clients or leads.
    - Follow-Ups: Follow-up meeting actions.
    - Attendance: Attendance tracking.

    **Instructions**:
    1. **Identify Topic**: Determine the main topic of the paragraph (1-3 words, e.g., 'Salesforce Automation', 'Inventory Management').
    2. **Split the Input**: If the paragraph contains multiple topics, questions, or answers, split it into separate entries. Each entry should focus on a single question and answer pair.
    3. **Extract Question and Answer**: Identify or infer the question and answer for each entry.
    4. **Select Category**: Choose the most relevant category from the list above.
    5. **Generate Keywords**: Extract 3–10 keywords for each entry.
    6. **Set Source**: Always set `source` to "faq".
    7. **Output**: Return a JSON object with a `topic` field and an `entries` array, each entry matching the schema above (without the `id` field).

    **Example Input**:
    "SFA stands for Salesforce Automation. It is a digital tool that helps sales teams manage their activities, including tracking client visits, scheduling meetings, recording expenses, and managing tasks. To log in to the SFA application, users need to enter their credentials and the Company Identifier."

    **Example Output**:
    ```json
    {{
        "topic": "Salesforce Automation",
        "entries": [
            {{
                "text": "Question: What is the purpose of the SFA application? Answer: The SFA application helps sales teams manage activities such as tracking client visits, scheduling meetings, recording expenses, and managing tasks.",
                "metadata": {{
                    "source": "faq",
                    "category": "Overview",
                    "keywords": ["SFA", "purpose", "salesforce", "tracking", "meetings", "expenses", "tasks"]
                }}
            }},
            {{
                "text": "Question: How do I log in to the SFA application? Answer: Enter your credentials and the Company Identifier to log in.",
                "metadata": {{
                    "source": "faq",
                    "category": "Login",
                    "keywords": ["login", "credentials", "Company Identifier", "SFA"]
                }}
            }}
        ]
    }}
    ```

    **Input Paragraph**:
    {paragraph}

    **Output**:
    Return a JSON object with a `topic` field and an `entries` array, each entry matching the schema above (without the `id` field).
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
        generated_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        return generated_text
    except requests.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "{}"

# Function to convert paragraph and append to doc1.json
def convert_paragraph_to_json(paragraph, json_file_path=FILE_PATH):
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDxYogQnjYmTbvghsqSWCKtot5DpygC8hA")
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
    last_id = max(int(item["id"]) for item in json_data) if json_data else 0
    
    # Call Gemini API to process the paragraph
    gemini_response = call_gemini_api(paragraph, api_key)

    def extract_json_from_response(response_text):
        code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
        if code_block_match:
            return code_block_match.group(1)
        array_match = re.search(r"(\{[\s\S]*\})", response_text)
        if array_match:
            return array_match.group(1)
        return response_text

    try:
        json_str = extract_json_from_response(gemini_response)
        parsed_response = json.loads(json_str)
        topic = parsed_response.get("topic", "Unknown Topic")
        entries = parsed_response.get("entries", [])
        if isinstance(entries, dict):
            entries = [entries]
    except Exception:
        print("Error: Gemini API did not return valid JSON")
        topic = "Unknown Topic"
        entries = [{
            "text": f"Question: Unknown Answer: Unable to process paragraph",
            "metadata": {
                "source": "faq",
                "category": "Overview",
                "keywords": ["error", "unknown"]
            }
        }]
    
    # Construct new JSON objects with unique IDs
    new_entries = []
    for entry in entries:
        last_id += 1
        new_entry = {
            "id": str(last_id),
            "text": entry.get("text", "Question: Unknown Answer: Unable to process paragraph"),
            "metadata": entry.get("metadata", {
                "source": "faq",
                "category": "Overview",
                "keywords": ["error", "unknown"]
            })
        }
        json_data.append(new_entry)
        new_entries.append(new_entry)
    
    # Write the updated JSON back to the file
    try:
        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Successfully appended {len(new_entries)} new entr{'y' if len(new_entries)==1 else 'ies'} to {json_file_path}")
    except Exception as e:
        print(f"Error writing to {json_file_path}: {e}")
    
    return {"topic": topic, "entries": new_entries}

# Example usage
if __name__ == "__main__":
    sample_paragraph = """
        SFA stands for Salesforce Automation. It is a digital tool that helps sales teams manage their activities, including tracking client visits, scheduling meetings, recording expenses, and managing tasks. The SFA application is designed to streamline the sales process and improve efficiency.
        To log in to the SFA application, users need to enter their credentials and the Company Identifier.
    """
    
    try:
        result = convert_paragraph_to_json(sample_paragraph)
        print("New entries added with topic:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing paragraph: {e}")