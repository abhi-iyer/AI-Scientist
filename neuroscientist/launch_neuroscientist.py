import json
import os
import os.path as osp
from typing import List, Dict
from tqdm import tqdm

from neuroscientist.generate_ideas import *
from neuroscientist.idea_reviewer import *
from neuroscientist.paper_writer import *

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

# configuration stuff
NUM_IDEAS = 10  # number of ideas to generate
NUM_ROUNDS_CONSISTENCY_CHECK = 5  # refinement iterations per idea
NUM_ROUNDS_NOVELTY_CHECK = 5  # number of rounds of novelty checking
NUM_ROUNDS_VALIDITY_CHECK = 5 # number of rounds of critique

SAVE_DIR = "./neuroscientist/"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-powered neuroscientist that generates unifying theories for neuroscience.")
    parser.add_argument("--model", type=str, default="deepseek-v3-openrouter", choices=AVAILABLE_LLMS, help="LLM to use")
    args = parser.parse_args()

    # create client
    client, client_model = create_client(args.model)
    
    # generate multi-level ideas and refine them for coherency
    # _ = generate_ideas(base_dir=SAVE_DIR, client=client, model=client_model, num_ideas=NUM_IDEAS, num_rounds_consistency=NUM_ROUNDS_CONSISTENCY_CHECK)

    # check novelty of each multi-level idea
    # _ = check_idea_novelty(base_dir=SAVE_DIR, client=client, model=client_model, num_rounds_novelty=NUM_ROUNDS_NOVELTY_CHECK)

    # _ = idea_reviewer(base_dir=SAVE_DIR, client=client, model=client_model, num_rounds_validity=NUM_ROUNDS_VALIDITY_CHECK)

    generate_paper(base_dir=SAVE_DIR, client=client, model=client_model)

    



