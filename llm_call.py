from text_extractor import DataExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity
from dotenv import load_dotenv
import os
import markdown
import re
from weasyprint import HTML
import matplotlib.pyplot as plt
import hashlib

load_dotenv()  

from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

def create_pdf_from_markdown(markdown_text, output_filename=""):
    """
    Convert markdown file to pdf for easy sharing and storage.
    Args: 
        markdown_text (str): The markdown content to convert.
        output_filename (str): The name of the output PDF file.
    """
    # CORRECTED: 'extensions' instead of 'extensions'
    html_text = markdown.markdown(
        markdown_text, extensions=['markdown.extensions.fenced_code']
    )
    
    # CORRECTED: Added f-string to print the filename
    print(f"Generating PDF: {output_filename}...")
    HTML(string=html_text, base_url=os.getcwd()).write_pdf(output_filename)
    print("PDF generated successfully.")

def insert_images_into_markdown(markdown_content, image_mapping):
    """
    Inserts image links into the markdown content before the figure description.

    Args:
        markdown_content (str): The summary text from the LLM.
        image_mapping (dict): A map of {figure_number (int): image_path (str)}.
    """
    def replacer(match):
        # The whole line that matched, e.g., "- **Figure 1:** Description..."
        original_line = match.group(0)
        # The figure number captured from the regex, e.g., '1'
        fig_num = int(match.group(2))
        
        image_path = image_mapping.get(fig_num)
        
        if image_path:
            # Create the markdown image tag and place it on the line above the description
            image_tag = f"![Figure {fig_num}]({image_path})"
            return f"{image_tag}\n{original_line}"
        else:
            # If no image is found for this figure number, return the line as is
            return original_line

    # This regex looks for lines starting with '*' or '-' followed by "**Figure <number>:**"
    # It captures the whole line (group 0), the bolded part (group 1), and the number (group 2)
    figure_line_pattern = re.compile(r"((?:[\*\-]\s*)?\*\*Figure (\d+):\*\*.*)", re.IGNORECASE)
    
    return figure_line_pattern.sub(replacer, markdown_content)
            

# LLM and Prompt setup
chat = ChatPerplexity(temperature=0, pplx_api_key=os.environ["PPLX_API_KEY"], model="sonar")
system = (
    "You are a helpful assistant that reads and summarizes academic research papers "
    "in a way that is accessible to a general audience. You extract key sections and explain technical content in simple terms."
)
human = r"""
Please summarize the following research paper content in a way that is clear, concise, and easily understood by a non-expert.

**Overall Instructions:**
- The summary title should be the paper's title, with the date of summarization as a subheading.
- When a key technical term is introduced (e.g., FPGA, Convolution, Protein Folding), you MUST provide a brief, parenthetical explanation of what it is. Also mark the term in **bold**.

Please break the summary into these specific sections using markdown:

1.  **Introduction / Abstract**: What is the core problem the paper addresses? What is its proposed solution and main finding?
2.  **Methodology**: How did the researchers conduct their work? Explain the key steps and techniques in a simple, logical sequence.
3.  **Theory / Mathematics**: Explain important theoretical concepts. After presenting the formula, then explain what it means and why it's used.
4.  **Key Diagrams or Visual Elements**: Describe each important figure or table. Start the description with '- **Figure X:**' and explain what the visual shows and its significance.
5.  **Conclusion**: What are the key results and takeaways? Crucially, end this section with a final sentence that explains the **"Why It Matters"** or the **"real-world impact"** of this research.

Text to summarize:
{input}
"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt | chat

# --- Workflow ---

# Step 1: Extract data from PDF
pdf_path = "/home/hari/computer_science/ai_projects/AI_Paper_Summarizer/Underwater Image Enhancement Using FPGA-Based Gaussian Filters.pdf"
extractor = DataExtractor(pdf_path)

paper_text = extractor.extract_text_from_pdf()
image_paths = extractor.extract_images_from_pdf() # Get the ordered list of image paths

# Step 2: Generate summary from LLM
if paper_text:
    response = chain.invoke({"input": paper_text})
    summary_content = response.content

    # Step 3: Create the mapping between figure numbers and image paths
    # Find all figure numbers mentioned in the summary text
    found_fig_numbers = re.findall(r'Figure (\d+)', summary_content, re.IGNORECASE)
    # Convert to unique integers and sort to maintain order
    unique_fig_numbers = sorted([int(n) for n in list(set(found_fig_numbers))])

    # Assume the order of extracted images matches the order of figures (1, 2, 3...)
    # Create a dictionary like {1: 'output_images/figure_1.png', 2: 'output_images/figure_2.png'}
    image_map = {fig_num: path for fig_num, path in zip(unique_fig_numbers, image_paths)}
    
    print("Figure-to-Image mapping created:", image_map)

    # Step 4: Insert images into the markdown summary
    final_markdown = insert_images_into_markdown(summary_content, image_map)
    
    # Step 5: Add a title and save the final markdown file
    final_markdown_with_title = f"# Summary of Research Paper\n\n{final_markdown}"

    with open('summary_new.md', 'w') as f:
        f.write(final_markdown_with_title)
    
    print("\nSuccessfully created summary.md with embedded images.")
else:
    print("Could not extract text from the PDF. Aborting.")

convert_to_pdf = input("Do you want to convert the markdown summary to PDF? (yes/no): ").strip().lower()
if convert_to_pdf == 'yes':
    output_pdf_filename = input("Enter pdf file name (without extension): ").strip() + ".pdf"
    
    # 2. Pass the new, PROCESSED variable to the PDF creator
    create_pdf_from_markdown(final_markdown_with_title, output_pdf_filename)
    