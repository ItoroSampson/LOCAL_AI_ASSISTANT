import time
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')

def full_performance_benchmark(prompt):
    print(f"--- 📊 Measuring TTFT, TPS, and TRL ---")
    
    start_time = time.perf_counter()
    ttft = None
    tokens_count = 0
    
    #  set stream=True to catch the VERY FIRST token
    stream = client.chat.completions.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0
    )
    
    print("Response: ", end="", flush=True)
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            # Capture TTFT on the very first piece of text
            if ttft is None:
                ttft = time.perf_counter() - start_time
            
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            tokens_count += 1
            
    end_time = time.perf_counter()
    
    # Final Calculations
    trl = end_time - start_time
    # Generation time is the total time minus the wait for the first token
    generation_time = trl - ttft
    tps = tokens_count / generation_time if generation_time > 0 else 0
    
    print(f"\n\n[DETAILED METRICS]")
    print(f"1. TTFT (Responsiveness): {ttft:.4f}s")
    print(f"2. TPS (Throughput):     {tps:.2f} tokens/sec")
    print(f"3. TRL (Total Latency):   {trl:.2f}s")

full_performance_benchmark("Summarize the importance of low latency in AI in 50 words.")