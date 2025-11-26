import json
from openai import OpenAI

from utils import md_to_text, clean_json_md_string

# %%
import os
from pathlib import Path

from dotenv import load_dotenv
env_file = 'settings.env'
dotenv_path = Path(env_file)
load_dotenv(dotenv_path=dotenv_path)






####################################
#        AI
####################################
# %%
# Send prompt to OpenRouter and return response
async def ai_openrouter(prompt=None, model="google/gemini-2.5-flash"):
    if not prompt:
        return {"status":"error", "reason": "OpenRouter AI request failed", "explanation": "Empty input"}
    # print(prompt)
    try:
    # if True:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        completion = client.chat.completions.create(
            model=model,
            extra_body={
                "models": ["openai/gpt-5-chat", "openai/gpt-oss-20b:free", "google/gemini-2.5-flash"],
            },
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart informations analyst and decision maker"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        # print(completion)
        if completion.choices[0].message.content is None:
            return {"status":"error", "reason": "OpenRouter AI request failed", "explanation": "No response received"}
        result_text = completion.choices[0].message.content
        # print(result_text)
        result_text = md_to_text(clean_json_md_string(result_text))
        result_json = json.loads(result_text)
        return result_json
    except Exception as e:
        return {"status":"error", "reason": "OpenRouter AI request failed", "explanation": str(e)}


# %%
# Generate AI prompt for outcomes
def generate_ai_prompt_outcomes(question="", rules="", outcomes=[], information=""):
    return f'# Task:\r\n\
Analyze given information, verify given question and rules, and decide relevant outcomes.\r\n\\r\n\
# Instructions:\r\n\
 - Read the information provided.\r\n\
 - Check if it is relevant to the given question and rules.\r\n\
 - Decide relevant outcomes exclusively from the list.\r\n\
 - Return the relevant outcome(s).\r\n\
# Output:\r\n\
Return the relevant outcome(s) in json, else return error with brief REASON.\r\n\
Example: `{json.dumps([{"outcome": "Lorem Ipsum", "id": "123456", "confidence": 0.5 }])}` or `{json.dumps({"error": "REASON"})}`\r\n\r\n\
# Question:\r\n\
**{json.dumps(question)}**\r\n\
# Rules:\r\n\
{json.dumps(rules)}\r\n\
# Possible outcomes:\r\n\
{json.dumps(outcomes)}\r\n\
# Information:\r\n\
{json.dumps(information)}'

# Generate AI prompt for outcomes
def generate_ai_prompt_sports_sides(teams=[], outcomes=[]):
    return f'# Task:\r\n\
Analyze given game outcomes (home, draw, away), compare and decide correct teams sides.\r\n\\r\n\
# Instructions:\r\n\
 - Read the provided game outcomes and check if it is relevant to given teams.\r\n\
 - Decide relevant outcomes and teams sides.\r\n\
 - Return the outcomes and teams sides in json.\r\n\
# Output:\r\n\
Return the same number of outcomes and teams sides in json, else return error with brief REASON.\r\n\
Example: `{json.dumps([{"outcome":"Team A","id":"111111","side":"home"},{"outcome":"Draw","id":"010101","side":"draw"},{"outcome":"Team B","id":"222222","side":"away"}])}` or `{json.dumps({"error": "REASON"})}`\r\n\r\n\
# Outcomes:\r\n\
{json.dumps(outcomes)}\r\n\
# Teams:\r\n\
{json.dumps(teams)}'


# %%
# TODO: Generate AI prompt for finding relevant questions
def generate_ai_prompt_questions(questions=[], information=""):
    return
