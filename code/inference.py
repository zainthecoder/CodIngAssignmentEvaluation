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
                "Your task is to assess the quality of solutions provided to specific questions and give a score between 1 and 10. "
                "Also, provide a brief feedback explaining the score."
            )
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
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
        fieldnames = ["question", "answer", "llm_score", "llm_feedback"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Wrap the QA data with tqdm for a progress bar
        for qa_pair in tqdm(qa_data, desc="Scoring solutions", unit="question"):
            question = qa_pair.get("question")
            answer = qa_pair.get("answer")

            # Construct the LLM prompt
            prompt = (
                f"Here is a question and its proposed solution:\n"
                f"Question: {question}\n"
                f"Solution: {answer}\n\n"
                "Please evaluate the solution on a scale of 1 to 10 and provide a brief feedback."
            )

            # Get evaluation from LLM
            llm_response = get_llm_response(prompt, model, tokenizer)

            # Extract score and feedback from the LLM response
            llm_score = ""
            llm_feedback = ""
            if "score:" in llm_response.lower():
                try:
                    score_start = llm_response.lower().find("score:") + len("score:")
                    score_end = llm_response.find("\n", score_start)
                    llm_score = llm_response[score_start:score_end].strip()
                except ValueError:
                    llm_score = "Unknown"

            if "feedback:" in llm_response.lower():
                try:
                    feedback_start = llm_response.lower().find("feedback:") + len("feedback:")
                    llm_feedback = llm_response[feedback_start:].strip()
                except ValueError:
                    llm_feedback = "No feedback provided."

            # Write to the output CSV
            writer.writerow({
                "question": question,
                "answer": answer,
                "llm_score": llm_score,
                "llm_feedback": llm_feedback
            })

    print(f"Solution scoring completed and saved to {output_file}.")
