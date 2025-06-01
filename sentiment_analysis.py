from openai import OpenAI
import os
import json
from tqdm import tqdm

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print ("Error: OpenAI API Key not provided.")

client = OpenAI(api_key=api_key)
    
function_def = {
    "name": "classify_sentiment",
    "description": "Classifies the sentiment of one or more German sentences related to migrants, refugees, or asylum seekers.",
    "parameters": {
        "type": "object",
        "properties": {
            "sentiments": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "irrelevant"]
                },
                "description": "The sentiment label for each input sentence."
                }
            },
        "required": ["sentiments"]
        }
    }
# initial prompt:
# prompt ="""You are a social scientist who is given sentences pertaining to refugees and assylum seekerrs. Your task is to classify the sentiment of each sentence as positive, negative, neutral, or irrelevant.
# If the sentence does not relate to refugees or asylum seekers, classify it as irrelevant.
# Return the output in the following exact format without any additional text or artifacts:
# {
#     "sentiments": [
#         "positive" | "negative" | "neutral" | "irrelevant"
#     ]
# }
# """

### Updated prompt that clearly defines the lables for better classification of sentiment towards refugees and asylum seekers:
"""
You are a social scientist studying how refugees, asylum seekers, and immigrants are portrayed in media texts.

Your task is to classify the **attitude or bias expressed *toward* these groups** in each sentence. Do not judge based on whether the situation is good or bad — instead, focus on how the groups are depicted.

Classify the sentiment as:
- "positive" → if the sentence portrays refugees/asylum seekers/immigrants in a positive or sympathetic light (e.g., deserving help, contributing to society, being welcomed)
- "negative" → if the sentence expresses fear, hostility, burden, blame, or other negative bias toward them
- "neutral" → if the sentence mentions them but in a neutral or factual way, without strong positive or negative connotation
- "irrelevant" → if the sentence is not about these groups at all

Return the output in the following exact format without any additional text or explanation:
{
    "sentiments": [
        "positive" | "negative" | "neutral" | "irrelevant"
    ]
}
"""


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def analyze_sentiment_batch(sentences, client):
    joined = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": joined}
        ],
        functions=[function_def],
        function_call={"name": "classify_sentiment"}
    )
    sentiments = response.choices[0].message.function_call.arguments
    return json.loads(sentiments)["sentiments"]


# File-level processing function
def process_file(input_path, output_path, batch_size=10):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    buffer = []

    for record in tqdm(data, desc=f"Processing {os.path.basename(input_path)}"):
        buffer.append(record)
        if len(buffer) >= batch_size:
            sentences = [r["sentence"] for r in buffer]
            sentiments = analyze_sentiment_batch(sentences, client)
            for r, s in zip(buffer, sentiments):
                r["sentiment"] = s
                results.append(r)
            buffer = []

    # Final batch
    if buffer:
        sentences = [r["sentence"] for r in buffer]
        sentiments = analyze_sentiment_batch(sentences, client)
        for r, s in zip(buffer, sentiments):
            r["sentiment"] = s
            results.append(r)

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    #years = list(range(2011, 2025)) 
    years = [2010]
    for year in years:
        file_name = f"deu_{year}"
        input_file = f"filtered_data/{file_name}.json"
        output_file = f"output/prompt2/sentiments_{file_name}.json"

        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
        else:
            process_file(input_file, output_file)
            print(f"Sentiment analysis completed. Results saved to {output_file}.")
