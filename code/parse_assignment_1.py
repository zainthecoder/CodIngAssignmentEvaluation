import json
from nbformat import read

# Path to your Jupyter Notebook file
input_file = "data/Assignment 1 (Deadline_ 14.11.2023, at 23_59)/Abgaben/Abedin_Zain_s28zabed_3823754/Assignment_1_ZainulAbedin_ShahzebQamar.ipynb"

output_file = "code/questions_and_answers.json"

# Load the Jupyter Notebook
with open(input_file, 'r', encoding='utf-8') as f:
    notebook_content = read(f, as_version=4)

# Initialize a list to store the extracted questions and answers
qa_dataset = []

# Temporary variable to hold the current question and its answers
current_question = None
current_answers = []

# Iterate through cells to extract questions and answers
for cell in notebook_content.cells:
    if cell.cell_type == "markdown":
        # Treat markdown cells as questions if they contain task descriptions
        if "Task" in cell.source or cell.source.startswith("###"):
            # If there's a previous question, finalize it
            if current_question:
                qa_dataset.append({
                    "question": current_question,
                    "answer": "\n".join(current_answers)
                })
            # Start a new question
            current_question = cell.source.strip()
            current_answers = []
    elif cell.cell_type == "code":
        # Append code cells as answers to the current question
        if current_question:
            current_answers.append(cell.source.strip())

# Finalize the last question-answer pair
if current_question:
    qa_dataset.append({
        "question": current_question,
        "answer": "\n".join(current_answers)
    })

# Save the extracted dataset to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(qa_dataset, f, indent=4)

print(f"Questions and answers have been saved to {output_file}")
