"""
Phase 3: Robust Inference Orchestrator
Implements a retry mechanism and graceful degradation for local LLM inference 
using Pydantic validation and Ollama.
"""
import json
import time
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')

class AIAnalysis(BaseModel):
    topic: str
    summary: str
    complexity_score: int

def robust_inference(prompt, max_retries=3):
    attempt = 0
    
    while attempt < max_retries:
        try:
            print(f"--- 🔄 Attempt {attempt + 1} ---")
            
            response = client.chat.completions.create(
                model="llama3.2:1b",
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No chatter. No stutter"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7 # Slight heat to get a different result on retry
            )
            
            # Try to validate
            raw_data = response.choices[0].message.content
            validated = AIAnalysis.model_validate_json(raw_data)
            return validated # ✅ Success! Exit the function.

        except Exception as e:
            attempt += 1
            print(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(1) 

    # --- 📉 GRACEFUL DEGRADATION ---
    print("\n[!] All retries failed. Reverting to Graceful Degradation Mode.")
    return AIAnalysis(
        topic="Error",
        summary="The AI was unable to generate a valid summary after multiple attempts.",
        complexity_score=0
    )


result = robust_inference("Analyze the impact of 5G on Cloud AI.")
print(f"\nFINAL RESULT: {result.summary}")