import streamlit as st
import json
from fileupdater import convert_paragraph_to_json, FILE_PATH

st.title("SFA FAQ JSON Updater")

st.write("""
Enter a paragraph about the SFA application.  
The app will extract Q&A, category, and keywords, and append it to the FAQ JSON file.
""")

paragraph = st.text_area("Input Paragraph", height=200)

if st.button("Process and Append"):
    if paragraph.strip():
        with st.spinner("Processing..."):
            try:
                result = convert_paragraph_to_json(paragraph)
                st.success("Entry successfully added to doc1.json!")
                st.subheader("Generated JSON Entry:")
                st.json(result)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a paragraph before submitting.")