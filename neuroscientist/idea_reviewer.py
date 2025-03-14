import json
import os
import os.path as osp
from typing import List, Dict
from tqdm import tqdm
import pyalex
from pyalex import Works

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers

validity_check_msg = "You are an experiment neuroscientist evaluating the validity of a neuroscience theory. You will have {num_rounds_validity} rounds to review and critique. You do not need to use them all."
validity_check_prompt = """You are an experiment neuroscientist evaluating the validity of a neuroscience theory.

This is round {current_round}/{num_rounds_validity}. 
Evaluate the following neuroscience theory for validity.

High-Level Theory JSON:
```json
{high_level_json}
```

Mid-Level Theory JSON: 
```json
{mid_level_json}
```

Low-Level Theory JSON:
```json
{low_level_json}
```

Determine whether:
1. There exists contradictory experimental neuroscience evidence that completely invalidates any 3 levels of the theory.
2. There are inconsistencies between the 3 levels that need to be resolved, where such inconsistencies are strongly grounded in neuroscience literature or experiments.

If you identify any issues, suggest modifications to the theory at any level.

Respond in the following format:
THOUGHT:
<THOUGHT>

UPDATED JSON:
```json
{{
  "High-Level": <UPDATED_HIGH_LEVEL_JSON>,
  "Mid-Level": <UPDATED_MID_LEVEL_JSON>,
  "Low-Level": <UPDATED_LOW_LEVEL_JSON>,
}}
```

Ensure that the JSON is well-formatted and parsable.
If no changes are needed, include "I am done" before the JSON and return the previous JSON structures EXACTLY.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES.
"""

# driver function for idea reviewer
def idea_reviewer(base_dir, client, model, num_rounds_validity):
    json_path = os.path.join(base_dir, "ideas.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                ideas = json.load(f)
            print(f"Loaded {len(ideas)} existing ideas from {json_path}")
        except json.JSONDecodeError:
            raise("Error: Could not parse existing ideas.json.")
    else:
        raise(f"Error: {json_path} not found.")


    # assert there exists at least one novel idea
    if not any(idea["novel"] for idea in ideas):
        raise ValueError("No novel ideas found.")
    

    idea_archive = []

    for idx, idea in enumerate(ideas):
        if not idea["novel"]:
            continue

        print(f"Reviewing idea {idx}")

        high_level, mid_level, low_level = idea["High-Level"], idea["Mid-Level"], idea["Low-Level"]
        msg_history = []

        for j in range(num_rounds_validity):
            text, msg_history = get_response_from_llm(
                validity_check_prompt.format(
                    current_round=j + 1,
                    num_rounds_validity=num_rounds_validity,
                    high_level_json=json.dumps(high_level, indent=2),
                    mid_level_json=json.dumps(mid_level, indent=2),
                    low_level_json=json.dumps(low_level, indent=2),
                ),
                client=client,
                model=model,
                system_message=validity_check_msg.format(num_rounds_validity=num_rounds_validity),
                msg_history=msg_history,
            )

            json_output = extract_json_between_markers(text)
            assert json_output, "Failed to extract JSON in validity check"

            # update the levels
            high_level, mid_level, low_level = json_output["High-Level"], json_output["Mid-Level"], json_output["Low-Level"]

            # done with changes
            if "I am done" in text:
                print(f"Refinement converged after {j + 1} rounds.")
                break

        idea_archive.append(
            {"High-Level": high_level, "Mid-Level": mid_level, "Low-Level": low_level}
        )
    
    # save reviewed ideas
    json_path = os.path.join(base_dir, "ideas.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idea_archive, f, indent=4)
    
    print(f"Saved validity-checked ideas to {json_path}")

    return idea_archive