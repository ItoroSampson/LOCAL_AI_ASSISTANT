import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')

# 1.  (Pydantic Model)
class AIAnalysis(BaseModel):
    topic: str = Field(description="The subject being discussed")
    summary: str = Field(description="A concise 15-word summary")
    complexity_score: int = Field(ge=1, le=10, description="Complexity from 1 to 10")

def run_structured_test():
    print("--- ⚡ Starting Structured JSON Inference ⚡ ---")
    
    #  System Message to 'prime' the model for JSON (GUARDRAILS)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
        {"role": "user", "content": "Analyze ' AI Engineering'. Provide 'topic', 'summary', and 'complexity_score'."}
    ]

    full_response = ""
    
    # Using stream=True so I can see the 'First Token'
    stream = client.chat.completions.create(
        model="llama3.2:1b",
        messages=messages,
        stream=True,
        response_format={"type": "json_object"},
        temperature=0
    )

    print("Live JSON Stream: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print("\n\n--- 🛡️ Starting Pydantic Validation ---")
    
    # 2. The Verification Logic
    try:
        # Convert the string into a dictionary
        json_data = json.loads(full_response)
        # Force the dictionary to follow  Pydantic rules..... i think i'm loving this pydantic 
        validated_data = AIAnalysis(**json_data)
        
        print("[✅ SUCCESS]: Data is structured and valid.")
        print(f"Topic: {validated_data.topic}")
        print(f"Summary: {validated_data.summary}")
        print(f"Complexity: {validated_data.complexity_score}/10")
        
    except json.JSONDecodeError:
        print("[❌ ERROR]: The AI did not produce valid JSON.")
    except ValidationError as e:
        print(f"[❌ ERROR]: JSON was valid, but didn't match our schema: {e}")

if __name__ == "__main__":
    run_structured_test()