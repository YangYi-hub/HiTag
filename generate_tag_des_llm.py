import json
import os
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(
    description='Generate LLM tag descriptions')
parser.add_argument('--model_path',
                    default='/path/to/Qwen2.5-72B-Instruct',
                    help='Path to the local LLM model')
parser.add_argument('--tag_list_path',
                    default='/path/to/tag_list.txt',
                    help='Path to the tag list file')
parser.add_argument('--output_file_path',
                    help='save path of llm tag descriptions',
                    default='/path/to/save')
parser.add_argument('--definition_file',
                    default='/path/to/tag_definition',
                    help='Path to the file that holds definitions for each tag in matching order')


def analyze_tags(tag, definition, llm, sampling_params):
    # Generate LLM tag descriptions
    llm_prompts = [
        f"Describe concisely what a(n) {tag} looks like:",
        f"How can you identify a(n) {tag} concisely?",
        f"What does a(n) {tag} look like concisely?",
        f"What are the identifying characteristics of a(n) {tag}:",
        f"Please provide a concise description of the visual characteristics of {tag}:"
    ]

    result_lines = []

    # The first line as a default short description
    result_lines.append(f"a photo of a {tag}.")
    result_lines.append(f"i took a picture : itap of a {tag}.")
    result_lines.append(f"pics : a bad photo of the {tag}.")
    result_lines.append(f"pics : a origami {tag}.")
    result_lines.append(f"pics : a photo of the large {tag}.")
    result_lines.append(f"pics : a {tag} in a video game.")
    result_lines.append(f"pics : art of the {tag}.")
    result_lines.append(f"pics : a photo of the small {tag}.")
    result_lines.append(f"Definition for '{tag}': {definition}")

    for llm_prompt in tqdm(llm_prompts, leave=False, desc=f"Processing {tag}"):
        system_prompt = ("You are a direct AI assistant describing visual traits.")
        
        user_prompt = (
            f"Here is the intended meaning of '{tag}': {definition}\n\n"
            f"{llm_prompt}"
        )
        '''
        #V6
        user_prompt = (
            f"Here is the intended meaning of '{tag}': {definition}\n\n"
            "Focus on describing the key visual traits"
            f"{llm_prompt}"
            )
        '''
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # Use the LLM to generate the response
        inputs = [{"prompt": prompt}]
        outputs = llm.generate(inputs, sampling_params)
        result_lines.append(outputs[0].outputs[0].text.strip())

    return result_lines


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading model...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=4,#8,
    )
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=77,
    )
    print("Model loaded successfully.")

    # Load tag definitions
    with open(args.definition_file, 'r', encoding='utf-8') as df:
        definitions = json.load(df)

    # Read categories
    tag_list_path = args.tag_list_path
    print(f"Reading categories from {tag_list_path}")
    with open(tag_list_path, 'r', encoding='utf-8') as f:
        categories = [line.strip() for line in f.readlines()]

    print(f"Found {len(categories)} categories")

    output_file_path = args.output_file_path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    tag_descriptions = {}
    cnt = 0
    for i, tag in enumerate(tqdm(categories, desc="Processing tags")):
        definition = definitions[i]  # Match by index
        tag_descriptions[tag] = analyze_tags(tag, definition, llm, sampling_params)
        cnt += 1

        if cnt % 20 == 0:
            print(f"[Checkpoint] Saving interim results after {cnt} tags...")
            with open(output_file_path, 'w') as w:
                json.dump(tag_descriptions, w, indent=3)

    print(f"Saving final descriptions to {output_file_path}")
    with open(output_file_path, 'w') as w:
        json.dump(tag_descriptions, w, indent=3)

    print("Done!")