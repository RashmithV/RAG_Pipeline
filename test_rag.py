from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_comp_Engg():
    assert query_and_validate(
        question="How many semesters does masters in Computational engineering has? )",
        expected_response="4 semesters",
    )


def test_Seminar_CE():
    assert query_and_validate(
        question="Is there a seminar in Computational Engineering Masters?)",
        expected_response="Yes",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    print(expected_response)
    #print(response_text)

    model = Ollama(model="mistral") #model 1
    #model = Ollama(model="llama2") #model 2
    #model = Ollama(model="gemma2") #model 3
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    
    

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
