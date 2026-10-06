import ollama
import json
from tqdm import tqdm
import re
import os

# extract JSON from a string, handling code fences if present
def extract_json(text):
    # strip ```json ... ``` or ``` ... ``` fences if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)

# load and clean prompts
with open("instructions.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open("example.txt", "r", encoding="utf-8") as f:
    example = f.read()

example = example.replace("\n", "")

system_prompt = system_prompt.replace("{example}", example)

with open("../data/corpse.jsonl", "r", encoding="utf-8") as f:
    documents = [json.loads(line) for line in f]

# main processing loop
NUM_CTX = 128000  # 128k context length
RESTART_THRESHOLD = 0.95
TEMPERATURE = 0.55
MODEL = "gemma4" 
MAX_ATTEMPT = 3
output_file = "../data/processed_corpse.jsonl"
messages = [{"role": "system", "content": system_prompt}]
results = []

def get_response(current_messages, content):
    current_messages.append({"role": "user", "content": content})
    response = ollama.chat(
        model=MODEL,  
        messages=current_messages,
        options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    )
    return response


for i,doc in enumerate(tqdm(documents)):

    attemp_count = 0
    while attemp_count < MAX_ATTEMPT:
        try:
            response = get_response(messages, doc["text"])
            reply = response["message"]["content"]

            # update message
            messages.append({"role": "assistant", "content": reply})
            # extract reply as json
            reply_dict = extract_json(reply)
            # append it to results
            results.append(reply_dict)
            break  

        except Exception as e:
            attemp_count += 1
            print(f"Error decoding JSON for document {i}, attempt {attemp_count}: {e}")
            if attemp_count >= MAX_ATTEMPT:
                print(f"Max attempts reached for document {i}. Skipping.")
                break

    # check context usage
    used = response["prompt_eval_count"] + response["eval_count"]
    pct = used / NUM_CTX
    if pct > RESTART_THRESHOLD:
        print(f"Context usage {pct:.2%} exceeded threshold. Restarting conversation.")
        messages = [{"role": "system", "content": system_prompt}]

    

    # create the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # write results to file every 20 documents being processed
    if (i + 1) % 20 == 0:
        with open(output_file, "a", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        # clear results after writing
        results = []
    

