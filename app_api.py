import streamlit as st
import requests
import json
import pandas as pd  # Import pandas for CSV conversion

# Flask API base URL
API_BASE_URL = "http://localhost:5556"

st.title("Test Flask API with Streamlit")

# Select API endpoint
endpoint = st.selectbox(
    "Select API Endpoint",
    ["/find/relevant/data/from/speech", "/generate/data"]
)

# Input fields for keys and summary
st.subheader("Input Data")
keys = st.text_area("Enter Keys (comma-separated)", "nda signed, contract value, legal review")
summary = st.text_area(
    "Enter Summary",
    "Legal team confirmed NDA was signed last week. Contract value proposed at $1.3M. "
    "Awaiting legal review from the client’s side. Compliance flagged a missing data processing clause. "
    "Final approval is pending from procurement."
)

# Convert keys to a list
keys_list = [key.strip() for key in keys.split(",") if key.strip()]

# Button to send request
if st.button("Send Request"):
    # Prepare the payload
    payload = {
        "keys": keys_list,
        "summary": summary
    }

    # Send the request to the selected endpoint
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload)
        if response.status_code == 200:
            st.success("Request Successful!")
            response_data = response.json()
            st.json(response_data)

            # Convert the response to a CSV format
            if isinstance(response_data, dict) and "data" in response_data:
                # Flatten the JSON data into a DataFrame
                data = response_data["data"]

                # Ensure all values are strings for CSV conversion
                flattened_data = {key: str(value) for key, value in data.items()}
                df = pd.DataFrame.from_dict(flattened_data, orient="index").reset_index()
                df.columns = ["Key", "Value"]

                # Convert DataFrame to CSV
                csv_data = df.to_csv(index=False)

                # Add a download button for the CSV response
                st.download_button(
                    label="Download Response as CSV",
                    data=csv_data,
                    file_name="response.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Response data is not in the expected format.")
        else:
            st.error(f"Error: {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"An error occurred: {e}")