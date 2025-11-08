import os, re, requests, base64, getpass, json, time

from openai import OpenAI
from libs.embedding import retrieve_similar, compute_save_embeddings

def get_prompt(usecase, project_description):
    return """You are a tester tasked with creating comprehensive test cases for a given usecase description.

## Project description
The project encompasses a comprehensive online educational platform designed for students seeking to enhance their learning experiences through various functionalities. Key features include account registration, course enrollment, participation in live classes, accessing recorded lectures and eBooks, taking quizzes, and viewing progress reports. The platform also facilitates personalized interactions through smart notes and guidelines for extra-curricular activities. With a focus on user engagement and academic support, the project incorporates multiple use cases that address both student and user requirements, ensuring that users can efficiently navigate their educational journey, manage personal information, and receive timely support. Through rigorous testing scenarios, the platform aims to provide a seamless and effective learning environment, accommodating the diverse needs of students and educators alike.

## Usecase description

{
    "name": "Changing Personal Information",
    "scenario": "A user wants to change or update his personal information",
    "actors": "User",
    "preconditions": "User must login to his account",
    "steps": [
        "User logs in to his account",
        "User navigates to his profile settings",
        "User clicks on the button to edit personal information",
        "User updates the personal information (i.e Name, Gender, Birthday, Class Shift, Institution, Guadian's Name, Guadian's Mobile Number)"
    ]
}

## Testcase

[
    {
        "name": "Successful Personal Information Update",
        "description": "Verify that a user can successfully update his personal information",
        "input": {
            "userId": "user_12345",
            "name": "John Doe",
            "gender": "Male",
            "birthday": "1990-01-01",
            "classShift": "Morning",
            "institution": "ABC School",
            "guardianName": "Jane Doe",
            "guardianMobile": "01712345678"
        },
        "expected": {
            "outcome": "Personal information update successful",
            "status": "Information Updated"
        }
    },
    {
        "name": "Failed Personal Information Update",
        "description": "Verify that a user cannot update his personal information if any of the provided information is empty",
        "input": {
            "userId": "user_12345",
            "name": "John Doe",
            "gender": null,
            "birthday": "1990-01-01",
            "classShift": "Morning",
            "institution": "ABC School",
            "guardianName": "Jane Doe",
            "guardianMobile": "01712345678"
        },
        "expected": {
            "outcome": "Personal information update failed",
            "status": "Incorrect Information"
        }
    }
]

## Project description
""" + project_description + """

## Usecase description
""" + usecase + """

## Testcase


--------
**Important Instruction:**
    - Understand the last usecase.
    - Generate test cases similar to the given example that covers both:
        - **Normal** and **Edge** case scenarios
        - **Positive** and **Negative** case scenarios
        - **Valid** and **Invalid** case scenarios
    - Do not add any explanation or any unnecessary word.
    - Your generated testcase must be json parsable and must follow the style of the given example.
"""

def parse_response(response: str) -> str:
    if response is None:
        return ''
    
    if "```" not in response:
        return response

    code_pattern = r'```((.|\n)*?)```'
    if "```json" in response:
        code_pattern = r'```json((.|\n)*?)```'

    code_blocks = re.findall(code_pattern, response, re.DOTALL)

    if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
        code_str = "\n".join(code_blocks[-1])
    elif type(code_blocks[-1]) == str:
        code_str = code_blocks[-1]
    else:
        code_str = response

    return code_str.strip()
    

def generate_testcases(usecase, proj_desc, client, embedding=False):
    """
    Generates structured software test cases from a given use case using OpenAI GPT-4o.
    Logs token usage, cost, and latency to stat.csv.
    """
 
    if embedding:
        # Retrieve similar examples in embeddings
        print("Embedding is retrieving ...")
        #compute_save_embeddings(usecase)
        retrieved = retrieve_similar(usecase, top_k=1)
        context = ""
        for text, typ, score in retrieved:
            context += text
    
    else:
        context = usecase
        print("No embedding retrieved")
        
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # OpenAI model name
        messages=[
            {"role": "system", "content": "You are an expert QA engineer generating well-structured software test cases in JSON format."},
            {"role": "user", "content": get_prompt(context, proj_desc)}
        ],
        temperature=0.0,
        top_p=0.95,
        max_tokens=2000
    )
    
    # print(f"Response {response}")

    end_time = time.perf_counter()

    # Extract text content and usage info
    message = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Estimated cost for GPT-4o (as of late 2024)
    cost = (2.5 * prompt_tokens + 10 * completion_tokens) / 1e6  # USD

    # Save statistics
    with open("statistics.csv", "a") as f:
        f.write(f"GPT-4o,{prompt_tokens},{completion_tokens},{cost:.6f},{end_time-start_time:.3f}\n")

    # Parse the model output
    return json.loads(parse_response(message))

