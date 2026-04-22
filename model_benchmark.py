import time
import csv
import json
import psutil
import os
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
MODELS = ["llama3.2:1b", "phi4-mini:latest", "mistral"] 
OUTPUT_FILE = "benchmark_results.csv"

# Load the 10 prompts
with open('prompts.json', 'r') as f:
    PROMPTS = json.load(f)

def get_memory_usage():
    # Gets the total system RAM usage in GB
    return psutil.virtual_memory().used / (1024**3)

def run_deep_benchmark(model_name, prompt_data):
    print(f"📊 Benchmarking {model_name}...")
    
    start_time = time.perf_counter()
    ttft = None
    tokens = 0
    full_text = ""
    
    # Capture RAM before model loads
    mem_before = get_memory_usage()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_data['prompt']}],
            stream=True,
            temperature=0
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                content = chunk.choices[0].delta.content
                full_text += content
                tokens += 1
        
        # Capture RAM during/after peak usage
        mem_after = get_memory_usage()
        trl = time.perf_counter() - start_time
        tps = tokens / (trl - ttft) if (trl - ttft) > 0 else 0

        return {
            "Model": model_name,
            "Category": prompt_data['category'],
            "TTFT": round(ttft, 3),
            "TPS": round(tps, 2),
            "TRL": round(trl, 2),
            "RAM_Delta_GB": round(mem_after - mem_before, 2),
            "Output": full_text[:50] + "..." # Snippet for verification
        }
    except Exception as e:
        return {"Model": model_name, "Status": f"Error: {e}"}


# Create the file and write header immediately
keys = ["Model", "Category", "TTFT", "TPS", "TRL", "RAM_Delta_GB", "Output"]
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()

# Main Loop with IMMEDIATE saving
for model in MODELS:
    print(f"--- Starting Model: {model} ---")
    for p_data in PROMPTS:
        res = run_deep_benchmark(model, p_data)
        
        # Append  single result to the CSV 
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerow(res)
        
        if "TTFT" in res:
            print(f"✅ {p_data['category']} | TPS: {res['TPS']} | RAM Delta: {res['RAM_Delta_GB']}GB")
        else:
            print(f"❌ Error on {p_data['category']}")

    print(f"❄️ Cooling down hardware for 30 seconds...")
    time.sleep(30) 

print(f"\n✨ DONE! Your full report data is safe in: {OUTPUT_FILE}")