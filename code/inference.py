import json
import csv
from config import get_reader_model, get_tokenizer
from tqdm import tqdm

# Function to generate LLM response for scoring solutions
def get_llm_response(prompt, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for technical solutions. "
                "Your task is to assess the quality of solutions provided to specific questions"
            )
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=2000, do_sample=True)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

if __name__ == "__main__":
    # Load the JSON data file
    input_file = "questions_and_answers.json"
    output_file = "scored_solutions.csv"

    # Prepare the model and tokenizer
    model = get_reader_model()
    tokenizer = get_tokenizer()

    # Load the questions and answers
    with open(input_file, "r", encoding="utf-8") as jsonfile:
        qa_data = json.load(jsonfile)

    # Open output CSV
    with open(output_file, "w", newline="", encoding="utf-8") as output:
        fieldnames = ["question", "answer", "llm_score", "llm_feedback", "file_name"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Wrap the QA data with tqdm for a progress bar
        for qa_pair in tqdm(qa_data, desc="Scoring solutions", unit="question"):
            question = qa_pair.get("question")
            answer = qa_pair.get("answer")
            file_name = qa_pair.get("file_name")
            
            # Extract total points from the question using regex
            import re
            points_match = re.search(r"\((\d+)\s*points\)", question)
            total_points = int(points_match.group(1)) if points_match else 10  # Default to 10 if no match
            
            # Construct the LLM prompt
            # Construct the LLM prompt with specified response format
            prompt = (
                f"Here is a question and its proposed solution:\n"
                f"Question: {question}\n"
                f"Solution: {answer}\n\n"
                f"Please evaluate the solution and provide your response in the following format:\n\n"
                f"Score: [Provide the score out of {total_points} points]\n"
                f"Feedback: [Provide brief feedback explaining the score]"
            )


            # Get evaluation from LLM
            llm_response = get_llm_response(prompt, model, tokenizer)
            print(llm_response)
            
            # Extract score and feedback from the LLM response
            llm_score = ""
            llm_feedback = ""

            # Extract the score using regex
            score_match = re.search(r"score:\s*(\d+)", llm_response, re.IGNORECASE)
            if score_match:
                llm_score = score_match.group(1).strip()
            else:
                llm_score = "Unknown"

            # Extract the feedback using regex
            feedback_match = re.search(r"feedback:\s*(.*)", llm_response, re.IGNORECASE | re.DOTALL)
            if feedback_match:
                llm_feedback = feedback_match.group(1).strip()
            else:
                llm_feedback = "No feedback provided."

            # Print for debugging
            print(f"Extracted Score: {llm_score}")
            print(f"Extracted Feedback: {llm_feedback}")

            # Write to the output CSV
            writer.writerow({
                "question": question,
                "answer": answer,
                "llm_score": llm_score,
                "llm_feedback": llm_feedback,
                "file_name": file_name
            })


    print(f"Solution scoring completed and saved to {output_file}.")
