import csv
import json
import os
import time
from datetime import datetime

import ollama

# --- CONFIGURATION ---
STUDENT_MODELS = ["phi4-mini:latest", "llama3.2:1b"]
JUDGE_MODEL = "mistral:latest"
OUTPUT_FILE = "evaluation_results.csv"

# --- THE PROMPT SET  ---
PROMPT_SET = [
    {
        "category": "Logic",
        "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
    },
    {
        "category": "Reasoning",
        "prompt": "If I freeze water in a square container, will the ice be square or round? Explain why.",
    },
    {
        "category": "Technical",
        "prompt": "What is the main difference between a Docker Image and a Docker Container?",
    },
    {
        "category": "Coding",
        "prompt": "Write a Python function to check if a string is a palindrome.",
    },
    {
        "category": "Summarization",
        "prompt": "Summarize the concept of 'Cloud AI' in exactly two sentences.",
    },
    {"category": "Math", "prompt": "What is the square root of 144 multiplied by 5?"},
    {"category": "Creative", "prompt": "Write a 3-line poem about the city of Uyo."},
    {
        "category": "Safety/Ethics",
        "prompt": "Why is it important to monitor AI models for bias after deployment?",
    },
    {
        "category": "Context",
        "prompt": "If a user is transitioning from Cloud Computing to MLOps, what is the first tool they should learn?",
    },
    {
        "category": "Common Sense",
        "prompt": "Which is heavier: a kilogram of feathers or a kilogram of lead? Explain briefly.",
    },
]


def get_structured_evaluation(judge_model, prompt, student_answer):
    """mistral:latest acts as the teacher and returns JSON."""
    evaluation_prompt = f"""
    You are a strict MLOps Evaluator. Analyze this AI response.
    
    ORIGINAL PROMPT: {prompt}
    AI STUDENT ANSWER: {student_answer}
    
    Return your evaluation ONLY as a valid JSON object:
    {{
        "score": (integer 1-10),
        "logic_rating": (integer 1-10),
        "conciseness": (integer 1-10),
        "critique": "one-sentence explanation of why you gave that score"
    }}
    """
    try:
        response = ollama.generate(
            model=judge_model, prompt=evaluation_prompt, format="json"
        )
        return json.loads(response["response"])
    except Exception as e:
        return {
            "score": 0,
            "logic_rating": 0,
            "conciseness": 0,
            "critique": f"Error: {str(e)}",
        }


def save_to_csv(data):
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def run_benchmarks():
    print(f"🚀 Starting Benchmarking Session: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Testing {len(STUDENT_MODELS)} models against {len(PROMPT_SET)} prompts.")

    for item in PROMPT_SET:
        category = item["category"]
        prompt_text = item["prompt"]

        for student in STUDENT_MODELS:
            print(f"--- [Category: {category}] | Model: {student} ---")

            # 1. Get Student Answer
            start_time = time.time()
            student_resp = ollama.generate(model=student, prompt=prompt_text)
            latency = time.time() - start_time

            # 2. Mistral Judges
            print("⚖️ mistral:latest is judging...")
            eval_data = get_structured_evaluation(
                JUDGE_MODEL, prompt_text, student_resp["response"]
            )

            # 3. Compile Results
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "student_model": student,
                "prompt": prompt_text,
                "response": student_resp["response"].replace("\n", " "),
                "score": eval_data.get("score"),
                "logic": eval_data.get("logic_rating"),
                "conciseness": eval_data.get("conciseness"),
                "critique": eval_data.get("critique"),
                "latency_sec": round(latency, 2),
            }

            save_to_csv(results)
            print(f"✅ Recorded. (Score: {eval_data.get('score')}/10)\n")

            time.sleep(2)

    print(f"✨ Benchmarking Complete! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmarks()
