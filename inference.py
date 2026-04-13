import time
from openai import OpenAI


client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama', 
)

print("---Starting Local AI Benchmark ---")
start = time.perf_counter()

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Write a short essay about the Future of AI Engineering ."}],
    temperature=0
)

end = time.perf_counter()
duration = end - start
tokens = response.usage.completion_tokens
tps = tokens / duration

print(f"\nAI: {response.choices[0].message.content}")
print(f"\n[STATS]")
print(f"Time: {duration:.2f}s | Tokens: {tokens} | Speed: {tps:.2f} tokens/sec")