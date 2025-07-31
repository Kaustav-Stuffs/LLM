import streamlit as st
import json
import os
from fileupdater import convert_paragraph_to_json, FILE_PATH

from dotenv import load_dotenv
import ast
import io
from PyPDF2 import PdfReader
import docx

# --- Load environment variables from .env if present ---
load_dotenv()

# --- Secure credential store using environment variable ---
CREDENTIALS = {}
admins_env = os.getenv("ADMINS_ENV")
if admins_env:
    try:
        CREDENTIALS = ast.literal_eval(admins_env)
    except Exception:
        st.error("Error parsing admin credentials from environment variable.")

def login():
    st.title("SFA FAQ Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    if login_btn:
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

st.title("SFA FAQ JSON Updater")

st.write(f"""
Welcome, **{st.session_state['username']}**!  
Enter a paragraph about the SFA application.  
The app will extract Q&A, category, and keywords, and append it to the FAQ JSON file.
You can edit the generated entry before saving, and manage your session entries before final update.
You can also edit or delete existing entries in the FAQ file.
""")

if "session_entries" not in st.session_state:
    st.session_state.session_entries = []

# --- File Upload Section ---
st.subheader("Or Upload a File (PDF, TXT, DOCX)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

if uploaded_file:
    file_text = ""
    if uploaded_file.type == "application/pdf":
        file_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        file_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        file_text = ""

    if file_text:
        st.text_area("Extracted Text", value=file_text, height=200, key="file_text_area")
        # Optionally, allow user to use this text as the paragraph input
        if st.button("Use Extracted Text as Input Paragraph"):
            st.session_state["paragraph"] = file_text
            st.rerun()

# Use session state for paragraph if set by file upload
if "paragraph" in st.session_state:
    paragraph = st.session_state["paragraph"]
    st.session_state.pop("paragraph")
else:
    paragraph = st.text_area("Input Paragraph", height=200)

if st.button("Generate JSON Entry"):
    if paragraph.strip():
        with st.spinner("Processing..."):
            try:
                result = convert_paragraph_to_json(paragraph)
                if isinstance(result, list):
                    st.session_state.session_entries.extend(result)
                else:
                    st.session_state.session_entries.append(result)
                st.success("Entry generated and added to session!")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a paragraph before submitting.")

st.subheader("Session Entries (Edit/Delete before saving to file)")

# Display and allow editing/deleting of session entries
for idx, entry in enumerate(st.session_state.session_entries):
    with st.expander(f"Entry {idx+1} (ID: {entry.get('id', 'N/A')})", expanded=True):
        # Editable fields
        text = st.text_area(f"Q&A Text (Entry {idx+1})", value=entry["text"], key=f"text_{idx}")
        category = st.text_input(f"Category (Entry {idx+1})", value=entry["metadata"]["category"], key=f"cat_{idx}")
        keywords = st.text_area(f"Keywords (comma-separated) (Entry {idx+1})", value=", ".join(entry["metadata"]["keywords"]), key=f"kw_{idx}")
        # Save edits
        if st.button(f"Update Entry {idx+1}", key=f"update_{idx}"):
            entry["text"] = text
            entry["metadata"]["category"] = category
            entry["metadata"]["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
            st.success(f"Entry {idx+1} updated!")
        # Delete option
        if st.button(f"Delete Entry {idx+1}", key=f"delete_{idx}"):
            st.session_state.session_entries.pop(idx)
            st.warning(f"Entry {idx+1} deleted!")
            st.rerun()

if st.session_state.session_entries:
    if st.button("Save All Session Entries to doc1.json"):
        try:
            # Load existing data
            file_path = FILE_PATH
            with open(file_path, "r") as f:
                data = json.load(f)
            # Append session entries
            data.extend(st.session_state.session_entries)
            # Write back to file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            st.success(f"Saved {len(st.session_state.session_entries)} entries to doc1.json!")
            st.session_state.session_entries = []
        except Exception as e:
            st.error(f"Error saving to file: {e}")

# --- Edit/Delete Existing Entries in doc1.json ---
st.header("Edit or Delete Existing Entries in doc1.json")

file_path = FILE_PATH
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            st.error(f"Error loading doc1.json: {e}")
            data = []
else:
    data = []

for idx, entry in enumerate(data):
    with st.expander(f"Entry {entry.get('id', idx+1)}", expanded=False):
        text = st.text_area(f"Q&A Text (ID {entry.get('id', idx+1)})", value=entry.get("text", ""), key=f"edit_text_{idx}")
        category = st.text_input(f"Category (ID {entry.get('id', idx+1)})", value=entry.get("metadata", {}).get("category", ""), key=f"edit_cat_{idx}")
        keywords = st.text_area(
            f"Keywords (comma-separated) (ID {entry.get('id', idx+1)})",
            value=", ".join(entry.get("metadata", {}).get("keywords", [])),
            key=f"edit_kw_{idx}"
        )

        if st.button(f"Save Changes (ID {entry.get('id', idx+1)})", key=f"save_{idx}"):
            data[idx]["text"] = text
            data[idx]["metadata"]["category"] = category
            data[idx]["metadata"]["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            st.success(f"Entry {entry.get('id', idx+1)} updated!")

        if st.button(f"Delete Entry (ID {entry.get('id', idx+1)})", key=f"del_{idx}"):
            data.pop(idx)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            st.warning(f"Entry {entry.get('id', idx+1)} deleted!")
            st.rerun()