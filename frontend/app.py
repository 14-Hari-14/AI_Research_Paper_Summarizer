import streamlit as st
import requests
import os
from datetime import datetime
from pathlib import Path
import re
import zipfile
import io

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI Research Paper Assistant")
BACKEND_URL = "http://127.0.0.1:8000"

# --- UI Layout ---
st.title("AI Research Paper Assistant 📄")

def render_summary_with_images(summary_markdown, file_name, base_url):
    file_name_stem = Path(file_name).stem
    image_pattern = re.compile(r'!\[\]\(([^)]+)\)')
    last_idx = 0
    for match in image_pattern.finditer(summary_markdown):
        st.markdown(summary_markdown[last_idx:match.start()], unsafe_allow_html=True)
        image_filename = match.group(1).strip()
        image_url = f"{base_url}/static/{file_name_stem}/{image_filename}"
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                st.image(image_url, use_column_width=True, caption=image_filename)
            else:
                st.warning(f"Image not found: {image_filename}")
        except requests.exceptions.RequestException as e:
            st.warning(f"Failed to load image {image_filename}: {e}")
        last_idx = match.end()
    if last_idx < len(summary_markdown):
        st.markdown(summary_markdown[last_idx:], unsafe_allow_html=True)

def create_markdown_zip(summary_markdown, file_name, image_filenames, backend_url):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr(f"{Path(file_name).stem}_summary.md", summary_markdown)
        file_name_stem = Path(file_name).stem
        for img in image_filenames:
            img_url = f"{backend_url}/static/{file_name_stem}/{img}"
            try:
                img_data = requests.get(img_url).content
                zip_file.writestr(img, img_data)
            except Exception as e:
                st.warning(f"Failed to include image {img} in ZIP: {e}")
    zip_buffer.seek(0)
    return zip_buffer

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
                markdown_url = ""
                pdf_url = ""

                for section_name, section_prompt in sections.items():
                    st.write(f"-> Generating section: {section_name}...")
                    
                    payload = {
                        "file_name": file_name,
                        "section_prompt": section_prompt,
                        "image_filenames": st.session_state.get('image_filenames', []) if section_name == "Key Diagrams or Visual Elements" else []
                    }
                    params = {"api_key": user_api_key}
                    try:
                        response = requests.post(f"{BACKEND_URL}/summarize/", json=payload, params=params)
                        if response.status_code == 200:
                            result = response.json()
                            summary_text = result['summary']
                            if section_name == "Key Diagrams or Visual Elements":
                                markdown_url = result.get('markdown_url', '')
                                pdf_url = result.get('pdf_url', '')
                            
                            lines = summary_text.strip().splitlines()
                            if lines and lines[0].strip('# ').strip().lower() == section_name.lower():
                                summary_text = '\n'.join(lines[1:]).strip()
                            
                            summary_parts.append(f"### {section_name}\n\n{summary_text}")
                        else:
                            summary_parts.append(f"### {section_name}\n\n*Error: {response.json().get('detail')}*")
                    except requests.exceptions.RequestException as e:
                        summary_parts.append(f"### {section_name}\n\n*Error: {e}*")

                st.session_state.final_summary = "\n\n".join(summary_parts)
                st.session_state.markdown_url = markdown_url
                st.session_state.pdf_url = pdf_url
            
        if 'final_summary' in st.session_state and st.session_state.final_summary:
            st.markdown("---")
            st.markdown("### Summary Preview")
            render_summary_with_images(st.session_state.final_summary, file_name, BACKEND_URL)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state.get('markdown_url'):
                    st.download_button(
                        label="Download Summary as Markdown",
                        data=requests.get(f"{BACKEND_URL}{st.session_state.markdown_url}").content,
                        file_name=f"{Path(file_name).stem}_summary.md",
                        mime="text/markdown",
                    )
                else:
                    st.warning("Markdown download unavailable")
            
            with col2:
                if st.session_state.get('pdf_url'):
                    st.download_button(
                        label="Download Summary as PDF",
                        data=requests.get(f"{BACKEND_URL}{st.session_state.pdf_url}").content,
                        file_name=f"{Path(file_name).stem}_summary.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.warning("PDF download unavailable")
            
            with col3:
                if st.session_state.get('image_filenames'):
                    st.download_button(
                        label="Download Markdown Summary as ZIP (with images)",
                        data=create_markdown_zip(st.session_state.final_summary, file_name, st.session_state.image_filenames, BACKEND_URL),
                        file_name=f"{Path(file_name).stem}_summary.zip",
                        mime="application/zip",
                    )
                else:
                    st.warning("ZIP download unavailable (no images)")

    with chat_tab:
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