import json
import csv
import re
import logging
from config import get_multiple_models
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("output.log", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Function to generate LLM response for scoring solutions
def get_llm_response(prompt, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for technical solutions. "
                "Your task is to assess the quality of solutions provided to specific questions."
            )
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    try:
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        generated_ids = model.generate(model_inputs, max_new_tokens=2000, do_sample=True)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    except Exception as e:
        logger.error(f"Error while generating response: {e}")
        return "Error: Unable to process response."

if __name__ == "__main__":
    input_file = "questions_and_answers.json"
    output_file = "scored_solutions.csv"
    k_passes = 3  # Number of iterations for averaging


    # Load model and tokenizer
    logger.info("Initializing model and tokenizer...")
    models = get_multiple_models()

    try:

        # Dynamically generate fieldnames based on model names
        individual_fields = []
        for model_name in models.keys():
            for curr_pass in range(k_passes):
                individual_fields.append(f"{model_name}_score_{curr_pass}_pass")
                individual_fields.append(f"{model_name}_feedback_{curr_pass}_pass")

        fieldnames = ["question", "answer", "file_name", "average_score"] + individual_fields

        # Load the questions and answers
        logger.info(f"Loading input data from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as jsonfile:
            qa_data = json.load(jsonfile)

        logger.info(f"Scoring solutions and saving to {output_file}...")
        with open(output_file, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            counter = 0
            for qa_pair in tqdm(qa_data, desc="Scoring solutions", unit="question"):
                question = qa_pair.get("question", "No question provided")
                answer = qa_pair.get("answer", "No answer provided")
                file_name = qa_pair.get("file_name", "Unknown file")

                # Extract total points
                points_match = re.search(r"\((\d+)\s*points\)", question)
                total_points = int(points_match.group(1)) if points_match else 10

                # Construct the LLM prompt
                prompt = (
                    f"Here is a question and its proposed solution:\n"
                    f"Question: {question}\n"
                    f"Solution: {answer}\n\n"
                    f"Please evaluate the solution and provide your response in the following format:\n\n"
                    f"Score: [Provide the score out of {total_points} points. please return only integer or fraction.]\n"
                    f"Feedback: [Provide brief feedback explaining the score]"
                )
                
                # Aggregation Multiple models
                results = {}
                model_scores = {model_name: 0 for model_name in models.keys()}
                model_feedbacks = {model_name: [] for model_name in models.keys()}


                # Perform K passes for each model
                for curr_pass in range(k_passes):
                    
                    for model_name, components in models.items():
                        
                        model = components['model']
                        tokenizer = components['tokenizer']
                        
                        # Generate LLM response
                        llm_response = get_llm_response(prompt, model, tokenizer)
                        logger.debug(f"Raw LLM response: {llm_response}")

                        print(llm_response)

                        # Extract score and feedback
                        score_match = re.search(r"score:\s*(\d+)", llm_response, re.IGNORECASE)
                        llm_score = score_match.group(1).strip() if score_match else "Unknown"

                        feedback_match = re.search(r"feedback:\s*(.*)", llm_response, re.IGNORECASE | re.DOTALL)
                        llm_feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."

                        # Append score and feedback
                        results[f"{model_name}_score_{curr_pass}_pass"] = int(llm_score)
                        results[f"{model_name}_feedback_{curr_pass}_pass"] = llm_feedback
                    
                
                avg_score=0 

                for model_name in models.keys():
                    for curr_pass in range(k_passes):
                        avg_score += results[f"{model_name}_score_{curr_pass}_pass"]
                
                avg_score = avg_score/(k_passes*len(models))
                # Add aggregated results to the dictionary
                results.update({
                    "question": question,
                    "answer": answer,
                    "file_name": file_name,
                    "average_score": avg_score,
                })

                # Write results to the CSV file
                writer.writerow(results)

                counter +=1
                if counter>3:
                    break

        logger.info(f"Scoring completed. Results saved to {output_file}.")
    except Exception as e:
        logger.error(f"An error occurred: {e}",exc_info=True)
