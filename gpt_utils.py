from openai import OpenAI
import os
import ast
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_option_parameters(user_input):
    prompt = f"""Extract option parameters from the text:
'{user_input}'
Return a Python dictionary with keys: S, K, T, r, sigma, option_type.
"""

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4-turbo" if needed
        messages=[
            {"role": "system", "content": "You are a financial assistant that extracts data for option pricing."},
            {"role": "user", "content": prompt}
        ]
    )

    return ast.literal_eval(response.choices[0].message.content)