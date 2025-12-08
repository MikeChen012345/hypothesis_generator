import json
import os
from workflow import AgentWorkflow
from llm import llm_workflow
from baseline import baseline_workflow
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from inference_auth_token import get_access_token
import logging_config
import apis

def load_dataset(path: str='dataset.json') -> list[dict]:
    """Load a dataset from a JSONL file.

    Args:
        path (str): Path to the JSONL file.
    Returns:
        list[dict]: List of data entries.
    """
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def get_output(model: str,
        dataset_path: str='dataset.json', 
        output_prefix: str='output', 
        ) -> list[dict]:
    """Load dataset and get output from the workflow.

    Args:
        model (str): The model to examine.
        dataset_path (str): Path to the dataset file.
        output_prefix (str): Prefix for the output files.
    Returns:
        list[dict]: List of output entries.
    """
    data = load_dataset(dataset_path)
    outputs = []
    failure = 0 # count of failures
    for i, entry in enumerate(data):
        print(f"Processing entry {i+1}/{len(data)}", flush=True, end='\r') # progress indicator
        initial_hypothesis = entry['hypothesis']
        while True:
            if model == "baseline":
                try:
                    final_hypothesis, rationale, citations, token_usage, elapsed_time = \
                        baseline_workflow(initial_hypothesis)
                    if final_hypothesis == "":
                        failure += 1
                    else:
                        break
                except Exception as e:
                    print(f"Error processing entry {i+1}: {e}", flush=True)
                    failure += 1
            elif model == "llm":
                try:
                    final_hypothesis, rationale, citations, token_usage, elapsed_time = \
                        llm_workflow(initial_hypothesis)
                    if final_hypothesis == "":
                        failure += 1
                    else:
                        break
                except Exception as e:
                    print(f"Error processing entry {i+1}: {e}", flush=True)
                    failure += 1
            elif model == "agent":
                workflow = AgentWorkflow(thread_id="thread_1", user_id="user_1")
                try:
                    final_hypothesis, rationale, citations, token_usage, elapsed_time = \
                        workflow.run_workflow(initial_hypothesis, show_message=False)
                    if final_hypothesis == "" or rationale == "":
                        failure += 1
                    else:
                        break
                except Exception as e:
                    print(f"Error processing entry {i+1}: {e}", flush=True)
                    failure += 1
            else:
                raise ValueError(f"Unknown model: {model}")
            
        output_entry = {
            'initial_hypothesis': initial_hypothesis,
            'final_hypothesis': final_hypothesis,
            'rationale': rationale,
            'citations': citations,
            'token_usage': token_usage,
            'elapsed_time': elapsed_time
        }
        outputs.append(output_entry)

        # Save outputs to file after each entry to avoid data loss in case of interruption
        with open(f"{output_prefix}_{model}.json", 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)

    return outputs


def get_dataset_info(dataset_path: str) -> dict:
    """Get information about the dataset.

    Args:
        dataset_path (str): Path to the dataset.
    Returns:
        dict: Information about the dataset.
    """
    data = load_dataset(dataset_path)
    num_entries = len(data)
    subjects = dict()
    divisions = dict()
    for entry in data:
        subject = entry.get('subject', 'Unknown')
        subjects[subject] = subjects.get(subject, 0) + 1
        division = entry.get('division', 'Unknown')
        divisions[division] = divisions.get(division, 0) + 1
    info = {
        'num_entries': num_entries,
        'subject_distribution': subjects,
        'division_distribution': divisions,
    }
    return info
    

def check_citation(citations: list[str]) -> bool:
    """
    Check if all citations are valid and exist.
    Args:
        citations (list[str]): List of citation strings.
    Returns:
        bool: True if all citations are valid, False otherwise.
    """
    for citation in citations:
        if not citation or not isinstance(citation, str):
            return False
        lst = citation.split('|')
        if len(lst) != 3: # Format error
            return False
        title, author, year = lst
        if not title.strip() or not author.strip() or not year.strip().isdigit():
            return False
        arxiv_result = apis.arxiv_paper_search("all:" + title, limit=5)
        semantic_scholar_result = apis.semantic_paper_search(title, limit=5)
        if title not in arxiv_result and title not in semantic_scholar_result:
            return False
        if author not in arxiv_result and author not in semantic_scholar_result:
            return False
        if year not in arxiv_result and year not in semantic_scholar_result:
            return False
    return True


def calculate_metrics(models: list[str], outputs: list[dict]=None, outputs_path: str="") -> dict:
    """Calculate metrics from the outputs.

    Args:
        models (list[str]): names of models used as the llm judges to evaluate the outputs.
        outputs (list[dict]): List of output entries.
        outputs_path (str): Path to the output file if outputs is not provided.
    Returns:
        dict: Calculated metrics.
    """
    if outputs is None and not outputs_path:
        raise ValueError("Either outputs or outputs_path must be provided.")
    
    if outputs is None and outputs_path:
        with open(outputs_path, 'r', encoding='utf-8') as f:
            outputs = json.load(f)
    
    total_token_usage = sum(entry['token_usage'] for entry in outputs)
    total_elapsed_time = sum(entry['elapsed_time'] for entry in outputs)
    num_entries = len(outputs)

    system_prompt = """
    You are an expert evaluator. Given the initial hypothesis and the final hypothesis, 
    evaluate the final hypothesis based on the following criteria:
    - Verifiability (whether supported by evidence)
    - Logical soundness (whether logically consistent)
    - Novelty (whether new or original)
    - Relevance to initial hypothesis
    - Clarity (whether clearly stated)
    Provide a score from 1 to 5 for each criterion, where 5 is the best.
    Provide your answer in the JSON format without any additional text:
    {
        "Verifiability": <score>,
        "Logical Soundness": <score>,
        "Novelty": <score>,
        "Relevance": <score>,
        "Clarity": <score>
    }

    If the final hypothesis is empty or is not a sentence in natural language, return 0 for all criteria.

    """

    model_config = logging_config.get_model_config()
    model_results = {}

    if not os.path.exists("results"):
        os.mkdir("results")

    # LLM judges evaluation
    for i, model_name in enumerate(models):
        base_url = model_config.get("base_url") or os.getenv("OPENAI_API_ENDPOINT")
        api_key = model_config.get("api_key") or os.getenv("OPENAI_API_KEY") or get_access_token()
        model = init_chat_model(
            model=model_name,
            model_provider="openai",
            temperature=0.5,
            timeout=30,
            max_retries=3,
            max_tokens=4096,
            base_url=base_url,
            api_key=api_key,
        )
        format_error = 0
        results = [[0, 0, 0, 0, 0] for _ in outputs] # entry_idx x criteria_idx
        for j, entry in enumerate(outputs):
            print(f"Evaluating entry {j+1}/{num_entries} with model {model_name}...", flush=True, end='\r')
            initial_hypothesis = entry['initial_hypothesis']
            final_hypothesis = entry['final_hypothesis']
            rationale = entry['rationale']
            citations = entry['citations']
            
            # Detect json format in the final hypothesis (which is wrong)
            if final_hypothesis.startswith("{") and final_hypothesis.endswith("}"):
                format_error += 1
                try:
                    final_hypothesis_json = json.loads(final_hypothesis)
                    final_hypothesis = final_hypothesis_json.get("final_hypothesis", "")
                except json.JSONDecodeError:
                    final_hypothesis = ""

            evaluation_prompt = f"""
            Initial Hypothesis: {initial_hypothesis}
            Final Hypothesis: {final_hypothesis}
            Rationale: {rationale}
            Citations: {', '.join(citations) if citations else 'None'}
            """
            round = 0
            max_retries = 5
            while round < max_retries:
                response = model.invoke([
                    SystemMessage(content=system_prompt), 
                    HumanMessage(content=evaluation_prompt)]).content
                try:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    response = response[start:end]
                    scores = json.loads(response)
                    for criterion, score in scores.items():
                        if criterion == "Verifiability":
                            results[j][0] = score
                        elif criterion == "Logical Soundness":
                            results[j][1] = score
                        elif criterion == "Novelty":
                            results[j][2] = score
                        elif criterion == "Relevance":
                            results[j][3] = score
                        elif criterion == "Clarity":
                            results[j][4] = score
                    break
                except json.JSONDecodeError:
                    print(f"Failed to parse evaluation response for entry {j+1} with model {model_name}: {response}", flush=True)  
        
                round += 1
        
        model_results[model_name] = {}
        for k in range(5):
            field_name = ["verifiability", "logical_soundness", "novelty", "relevance", "clarity"][k]
            avg_score = sum(results[j][k] for j in range(num_entries)) / num_entries if num_entries > 0 else 0 
            model_results[model_name][field_name] = avg_score
        
        model_results[model_name]['format_error'] = format_error

        with open(f"results/metrics.json", 'w', encoding='utf-8') as f:
            json.dump(model_results, f, ensure_ascii=False, indent=4)

        print("--------------------", flush=True)  # to move to the next line after progress indicator

    # check citation validity
    citation_validity = 0
    total_citations = 0
    for i, entry in enumerate(outputs):
        print(f"Evaluating entry {i+1}/{num_entries} on citation validity...", flush=True, end='\r')
        citations = entry['citations']
        if citations == []: # ignore empty citations (no citations provided)
            continue
        if check_citation(citations):
            citation_validity += 1
        total_citations += 1
    citation_validity_rate = citation_validity / total_citations if total_citations > 0 else 0

    # summarize metrics
    metrics = {
        'num_entries': num_entries,
        'average_token_usage': total_token_usage / num_entries if num_entries > 0 else 0,
        'average_elapsed_time': total_elapsed_time / num_entries if num_entries > 0 else 0,
        'format_error_rate': sum(model_results[model_name].get('format_error', 0) for model_name in model_results) / (num_entries * len(models)) if num_entries > 0 else 0,
        'citation_validity_rate': citation_validity_rate,
        'verifiability': sum(model_results[model_name]["verifiability"] for model_name in model_results) / len(model_results) if len(model_results) > 0 else 0,
        'logical_soundness': sum(model_results[model_name]["logical_soundness"] for model_name in model_results) / len(model_results) if len(model_results) > 0 else 0,
        'novelty': sum(model_results[model_name]["novelty"] for model_name in model_results) / len(model_results) if len(model_results) > 0 else 0,
        'relevance': sum(model_results[model_name]["relevance"] for model_name in model_results) / len(model_results) if len(model_results) > 0 else 0,
        'clarity': sum(model_results[model_name]["clarity"] for model_name in model_results) / len(model_results) if len(model_results) > 0 else 0,
    }
    with open(f"results/metrics_summary.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate metrics for hypothesis generation.")
    parser.add_argument('--model', type=str, required=True, help='model to be measured')
    parser.add_argument('--question_path', type=str, default='data/dataset.json', help='Path to the dataset file.')
    args = parser.parse_args()
    get_output(model=args.model, 
              output_prefix=f"results/output",
              dataset_path=args.question_path)
    
    # print(get_dataset_info('data/dataset.json'))

    judge_models = ["openai/gpt-oss-120b",
              "meta-llama/Meta-Llama-3.1-70B-Instruct",
              "meta-llama/Llama-4-Scout-17B-16E-Instruct",
              "google/gemma-3-27b-it",
              ]
    calculate_metrics(models=judge_models, outputs_path=f"results/output_{args.model}.json")