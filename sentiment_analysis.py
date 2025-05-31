from openai import OpenAI
import os
import json

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print ("Error: OpenAI API Key not provided.")

client = OpenAI(api_key=api_key)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def analyze_sentiment(sentence):

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
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a social scientist who given a sentence pertaining to immigration and refugees will classify it as positive negative or neutral."},
            {"role": "user", "content": f"""Analyze the sentiment of this German sentence: '{sentence}'
             Please classify it as positive, negative, neutral given the sentiment towards refugees, immigrants, asylum seekers and so on. If the sentence does not relate to refugees, immigrants, asylum seekers then classify it as irrelvant. 
            Return the output in the following exact format without any additional text or artifacts:
            {{
            "sentiment": "positive" | "negative" | "neutral" | "irrelevant",
            }}
             """}
        ],
        functions=[function_def],
        function_call={"name": "classify_sentiment"}
    )
    sentiment = response.choices[0].message.function_call.arguments
    return sentiment

output = analyze_sentiment("500 Menschen in Äthiopien vertrieben")

print(output)


