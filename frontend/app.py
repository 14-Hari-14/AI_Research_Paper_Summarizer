import streamlit as st
import requests
import os
from datetime import datetime
from pathlib import Path
import re

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI Research Paper Assistant")
BACKEND_URL = "http://127.0.0.1:8000"

# --- UI Layout ---
st.title("AI Research Paper Assistant 📄")

def render_summary_with_images(summary_markdown, file_name, base_url):
    file_name_stem = Path(file_name).stem
    # This regex now correctly finds the (see: figure_1.png) pattern
    image_pattern = re.compile(r'\(see:\s*([a-zA-Z0-9_]+\.(?:png|jpg|jpeg|gif))\)')
    
    # Split the text by the image references
    parts = image_pattern.split(summary_markdown)
    
    for i, part in enumerate(parts):
        # Odd-indexed parts are the captured image filenames
        if i % 2 == 1:
            image_filename = part.strip()
            image_url = f"{base_url}/static/{file_name_stem}/{image_filename}"
            # Display the image
            #st.write(f"Attempting to load image from: {image_url}") 
            st.image(image_url, use_container_width=True, caption=image_filename)
        # Even-indexed parts are the text in between
        else:
            if part.strip():
                st.markdown(part, unsafe_allow_html=True)
                
# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")
    user_api_key = st.text_input("Enter your Google AI API Key:", type="password", help="Your key is not stored.")
    uploaded_file = st.file_uploader("Upload your Research Paper (PDF):", type="pdf")

    if uploaded_file is not None and user_api_key:
        if st.button("Process Paper"):
            with st.spinner("Processing document... This may take a moment."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"api_key": user_api_key}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload-and-process/", files=files, params=params)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.processed_file_name = uploaded_file.name
                        st.session_state.image_filenames = result.get("image_filenames", [])
                        st.session_state.paper_title = os.path.splitext(uploaded_file.name)[0].replace('_', ' ').title()
                        st.success("✅ Paper processed successfully!")
                    else:
                        st.error(f"Error processing file: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection to backend failed: {e}")

# --- Main App Logic ---
if 'processed_file_name' not in st.session_state:
    st.info("👋 Welcome! Please provide your API key, upload a PDF, and click 'Process Paper' to begin.")
else:
    file_name = st.session_state.processed_file_name
    st.success(f"Ready to analyze **{file_name}**")
    
    summary_tab, chat_tab = st.tabs(["📊 Structured Summary", "💬 Chat with Document"])

    with summary_tab:
        st.header("One-Click Structured Summary")
        if st.button("Generate Full Summary", key="gen_summary"):
            with st.spinner("🚀 Generating summary section by section..."):
                base_instruction = "When a key technical term is introduced (e.g., FPGA), you MUST provide a brief, parenthetical explanation and mark the term in **bold**. Before starting the section dont write 'Based ONLY on the provided context' line just start with the content"
                sections = {
                    "Introduction / Abstract": f"Provide a summary covering the core problem, proposed solution, main findings, and any tradeoffs mentioned. {base_instruction}",
                    "Methodology": f"Describe the methodology, explaining the key steps and techniques in a simple sequence. {base_instruction}",
                    "Theory / Mathematics": f"Explain the core theoretical and mathematical principles. Format any formulas with LaTeX (e.g., $$...$$) and explain them. {base_instruction}",
                    "Key Diagrams or Visual Elements": f"Describe the paper's most important visual elements (figures, tables). If there are any visuals providing comparisons or conclusions make sure to add them. {base_instruction}",
                    "Conclusion": f"Summarize the key results and takeaways. Crucially, end with a sentence explaining the 'Why It Matters' or 'real-world impact'. {base_instruction}"
                }
                
                summary_parts = []
                title = st.session_state.get('paper_title', "Research Paper Summary")
                date_str = datetime.now().strftime("%B %d, %Y")
                summary_parts.append(f"# {title}\n*Summary Date: {date_str}*")

                for section_name, section_prompt in sections.items():
                    st.write(f"-> Generating section: {section_name}...")
                    
                    payload = {"file_name": file_name, "section_prompt": section_prompt}
                    if section_name == "Key Diagrams or Visual Elements":
                        payload["image_filenames"] = st.session_state.get('image_filenames', [])

                    params = {"api_key": user_api_key}
                    try:
                        response = requests.post(f"{BACKEND_URL}/summarize/", json=payload, params=params)
                        if response.status_code == 200:
                            summary_text = response.json()['summary']
                            
                            # --- FIX FOR REPEATING HEADINGS ---
                            # Clean the AI's response if it repeats the heading
                            # This removes the first line if it's identical to the section name
                            lines = summary_text.strip().splitlines()
                            if lines and lines[0].strip('# ').strip().lower() == section_name.lower():
                                summary_text = '\n'.join(lines[1:]).strip()
                            
                            summary_parts.append(f"### {section_name}\n\n{summary_text}")
                        else:
                            summary_parts.append(f"### {section_name}\n\n*Error: {response.json().get('detail')}*")
                    except requests.exceptions.RequestException as e:
                         summary_parts.append(f"### {section_name}\n\n*Error: {e}*")

                st.session_state.final_summary = "\n\n".join(summary_parts)
            
        if 'final_summary' in st.session_state and st.session_state.final_summary:
            st.markdown("---")
            st.markdown("### Summary Preview")
            # Use the corrected rendering function
            render_summary_with_images(st.session_state.final_summary, file_name, BACKEND_URL)
            
            st.download_button(
                label="Download Summary as Markdown",
                data=st.session_state.final_summary,
                file_name=f"{Path(file_name).stem}_summary.md",
                mime="text/markdown",
            )

    # --- Chat with Document Tab (Unchanged) ---
    with chat_tab:
        # This section remains the same as before
        st.header("Interactive Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Finding answer..."):
                    payload = {"file_name": file_name, "query": prompt}
                    params = {"api_key": user_api_key}
                    try:
                        response = requests.post(f"{BACKEND_URL}/chat/", json=payload, params=params)
                        if response.status_code == 200:
                            answer = response.json()['answer']
                        else:
                            answer = f"Error: {response.json().get('detail')}"
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection to backend failed: {e}")
                st.session_state.messages.append({"role": "assistant", "content": answer})