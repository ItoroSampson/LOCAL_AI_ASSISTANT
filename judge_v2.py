import csv
import json
import os
import time
from datetime import datetime

import ollama

# --- CONFIGURATION ---
# We are specifically focusing on Mistral being the student and Phi-4 being the judge
STUDENT_MODELS = ["mistral:latest"]
JUDGE_MODEL = "phi4-mini:latest"
OUTPUT_FILE = "evaluation_results.csv"

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


def get_cot_evaluation(judge_model, prompt, student_answer):
    """
    Forces the judge to use Chain of Thought reasoning
    before providing a final score.
    """
    cot_prompt = f"""
    You are a highly analytical AI Critic. You must evaluate the student's answer based on the original prompt.
    
    STEP 1: Analyze the prompt's requirements.
    STEP 2: Identify any factual or logical errors in the student's answer.
    STEP 3: Evaluate the tone and conciseness.
    STEP 4: Provide a final score.

    ORIGINAL PROMPT: {prompt}
    STUDENT ANSWER: {student_answer}
    
    Return your response ONLY as a valid JSON object:
    {{
        "thinking": "Your detailed step-by-step reasoning and critique goes here",
        "score": (integer 1-10),
        "logic_rating": (integer 1-10),
        "conciseness": (integer 1-10)
    }}
    """
    try:
        # Using format='json' ensures Phi-4 follows the structure
        response = ollama.generate(model=judge_model, prompt=cot_prompt, format="json")
        return json.loads(response["response"])
    except Exception as e:
        return {
            "thinking": f"Error: {str(e)}",
            "score": 0,
            "logic_rating": 0,
            "conciseness": 0,
        }


def save_to_csv(data):
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def run_benchmarks_v2():
    print("🚀 Starting v2 (Chain of Thought) Benchmark...")
    print(f"Judge: {JUDGE_MODEL} | Student: {STUDENT_MODELS[0]}")

    for item in PROMPT_SET:
        category = item["category"]
        prompt_text = item["prompt"]

        for student in STUDENT_MODELS:
            print(f"--- Analyzing Category: {category} ---")

            # 1. Get Mistral's Answer
            start_time = time.time()
            student_resp = ollama.generate(model=student, prompt=prompt_text)
            latency = time.time() - start_time

            # 2. Get Phi-4's CoT Grade
            print("🧠 Phi-4 is performing Chain of Thought evaluation...")
            eval_data = get_cot_evaluation(
                JUDGE_MODEL, prompt_text, student_resp["response"]
            )

            # 3. Compile Results
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "student_model": student,
                "judge_model": JUDGE_MODEL,
                "prompt": prompt_text,
                "response": student_resp["response"].replace("\n", " "),
                "thinking": eval_data.get("thinking").replace("\n", " "),
                "score": eval_data.get("score"),
                "latency_sec": round(latency, 2),
            }

            save_to_csv(results)
            print(f"✅ Grade Recorded. (Score: {eval_data.get('score')}/10)\n")

            time.sleep(3)

    print(f"✨ v2 Benchmarking Complete! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmarks_v2()
