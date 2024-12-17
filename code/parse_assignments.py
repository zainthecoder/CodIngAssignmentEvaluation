import os
import json
from nbformat import read
import re

# Define the base directory containing all assignment folders
base_dir = "data"
output_file = "code/questions_and_answers.json"

# Initialize a list to store the extracted questions and answers
qa_dataset = []

# Function to remove images from markdown content
def remove_images_from_markdown(markdown_text):
    # Remove Markdown-style images: ![alt](url)
    markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    # Remove Markdown images with base64-encoded data
    markdown_text = re.sub(r'!\[.*?\]\(data:image\/[a-zA-Z]+;base64,.*?\)', '', markdown_text)
    # Remove HTML-style images: <img src="url" ... />
    markdown_text = re.sub(r'<img.*?>', '', markdown_text)
    # Strip excessive blank lines after removing content
    return "\n".join([line for line in markdown_text.splitlines() if line.strip()])

# Function to check if a markdown cell is introductory
def is_introductory_cell(markdown_text):
    # Check for patterns indicating an introduction
    if re.match(r'^#\s+Introduction to .*?: Assignment \d+', markdown_text.strip()):
        return True
    if "In this exercise" in markdown_text or "we'll practice" in markdown_text:
        return True
    return False

# Function to process a single notebook and extract Q&A
def process_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = read(f, as_version=4)

    current_question = None
    current_answers = []
    temp_dataset = []
    skip_code_cell = False  # Flag to skip code cells after introductory markdown

    for cell in notebook_content.cells:
        if cell.cell_type == "markdown":
            clean_content = remove_images_from_markdown(cell.source)
            # Check if the cell is introductory, if so, skip it
            if is_introductory_cell(clean_content):
                skip_code_cell = True
                current_question = None  # Clear the current question
                continue  # Skip this markdown cell

            # Treat markdown cells as questions if they contain task descriptions
            if "Task" in clean_content or clean_content.startswith("###"):
                skip_code_cell = False  # Reset flag since it's a valid question
                if current_question:
                    temp_dataset.append({
                        "file_name": os.path.basename(notebook_path),
                        "question": current_question,
                        "answer": "\n".join(current_answers)
                    })
                current_question = clean_content.strip()
                current_answers = []
        elif cell.cell_type == "code":
            # Skip code cells if flagged by an introductory markdown
            if skip_code_cell:
                continue
            # Append code cells as answers to the current question
            if current_question:
                current_answers.append(cell.source.strip())

    # Finalize the last question-answer pair
    if current_question:
        temp_dataset.append({
            "file_name": os.path.basename(notebook_path),
            "question": current_question,
            "answer": "\n".join(current_answers)
        })

    return temp_dataset

# Walk through all assignment folders and process notebooks
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".ipynb"):  # Only process Jupyter Notebook files
            notebook_path = os.path.join(root, file)
            print(f"Processing: {notebook_path}")
            qa_dataset.extend(process_notebook(notebook_path))

# Save the combined Q&A dataset to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(qa_dataset, f, indent=4)

print(f"Questions and answers have been saved to {output_file}")
