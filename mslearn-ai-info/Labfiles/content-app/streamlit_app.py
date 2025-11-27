import streamlit as st
import requests
import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("ENDPOINT").rstrip("/")
KEY = os.getenv("KEY")
ANALYZER = os.getenv("ANALYZER_NAME")
API_VERSION = "2025-05-01-preview"


st.set_page_config(
    page_title="AI Content Understanding",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.result-card {
    background: #1f2937;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
}
.field-title {
    font-weight: bold;
    color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Content Understanding App")
st.caption("Smart extraction powered by Azure Content Understanding")

st.info(f"Active Analyzer: {ANALYZER}")

uploaded_file = st.file_uploader(
    "Upload a document, image, audio or video",
    type=["pdf", "jpg", "jpeg", "png", "mp3", "mp4"]
)


def analyze_file(file_bytes):
    url = f"{ENDPOINT}/contentunderstanding/analyzers/{ANALYZER}:analyze?api-version={API_VERSION}"

    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(url, headers=headers, data=file_bytes)
    response.raise_for_status()

    operation_url = response.headers["Operation-Location"]

    while True:
        result = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": KEY}).json()
        status = result.get("status")

        if status == "Succeeded":
            return result
        if status == "Failed":
            return {"error": "Analysis failed"}

        time.sleep(2)


def display_fields(data):
    for content in data["result"]["contents"]:
        if "fields" not in content:
            continue

        for field, field_data in content["fields"].items():
            value = "N/A"

            if field_data["type"] == "string":
                value = field_data["valueString"]
            elif field_data["type"] == "number":
                value = field_data["valueNumber"]
            elif field_data["type"] == "integer":
                value = field_data["valueInteger"]
            elif field_data["type"] == "array":
                value = ", ".join(map(str, field_data["valueArray"]))

            st.markdown(f"""
            <div class="result-card">
                <span class="field-title">{field}</span><br>
                {value}
            </div>
            """, unsafe_allow_html=True)


if uploaded_file:
    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing content..."):
            result = analyze_file(uploaded_file.read())

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("✅ Analysis Complete")

            tabs = st.tabs(["📋 Structured View", "🧾 Raw JSON"])

            with tabs[0]:
                display_fields(result)

            with tabs[1]:
                st.json(result)

            st.download_button(
                "⬇ Download Result JSON",
                data=json.dumps(result, indent=4),
                file_name="analysis_result.json",
                mime="application/json"
            )
