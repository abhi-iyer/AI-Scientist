import json
import os
import os.path as osp
from typing import List, Dict
from tqdm import tqdm
import pyalex
from pyalex import Works

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers



# highest level of abstraction
high_level_msg = "You are a theoretical neuroscientist developing unifying models of brain function."
high_level_prompt = """You are a theoretical neuroscientist developing unifying models of brain function.
Generate a new, high-level theory explaining how the neocortex processes information.
This theory should be inspired by first principles (e.g., physics, dynamical systems, information theory).
Avoid specific neural circuit details for now.

Respond in the following format:
THOUGHT:
<THOUGHT>

NEW THEORY JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the theory.
Explain how it unifies neuroscience perspectives.

In <JSON>, provide the new theory with these fields:
- "Name": A shortened descriptor of the theory. Lowercase, no spaces, underscores allowed.
- "Title": A full title summarizing the theory.
- "Description": A 2-3 sentence explanation of the theory.
- "Significance": A rating from 1 to 10 (lowest to highest).

Here are the previous high-level theories you've generated:
{previous_ideas}
"""

mid_level_msg = "You are a computational neuroscientist designing models of cortical computation."
mid_level_prompt = """You are a computational neuroscientist designing models of cortical computation.
Given the high-level theory: 
"{high_level_theory}",
convert it into a functional model of neocortical computation. 
Consider frameworks such as predictive coding, reservoir computing, or attractor networks.

Respond in the following format:
THOUGHT:
<THOUGHT>

NEW MODEL JSON:
```json
<JSON>
```

In <THOUGHT>, first analyze the implications of the theory for cortical computation.
Justify your choice of computational model.

In <JSON>, provide the mid-level framework with these fields:
- "Name": A shortened descriptor of the computational model.
- "Title": A full title summarizing the model.
- "Description": A 2-3 sentence explanation of the model.
- "Theoretical_Basis": The neuroscience principles supporting this model.
- "Relation_to_Theory": How this model implements the high-level theory.
- "Feasibility": A rating from 1 to 10.

Here are the previous high and mid-level theories you've generated:
{previous_ideas}
"""

low_level_msg = "You are a neuroscientist studying synaptic learning rules and cortical circuits."
low_level_prompt = """You are a neuroscientist studying synaptic learning rules and cortical circuits.
Given the computational framework: 
"{mid_level_model}", 
derive concrete neural mechanisms that could implement this model in the cortex.
Specify synaptic plasticity, inhibitory/excitatory interactions, and real-time learning mechanisms.

Respond in the following format:
THOUGHT:
<THOUGHT>

NEW MECHANISM JSON:
```json
<JSON>
```

In <THOUGHT>, explain the constraints and biological plausibility of the learning rule.

In <JSON>, provide the low-level mechanism with these fields:
- "Name": A shortened descriptor of the mechanism.
- "Title": A full title summarizing the mechanism.
- "Description": A 2-3 sentence explanation.
- "Biological_Basis": How this mechanism maps to experimental neuroscience findings.
- "Relation_to_Model": How this mechanism implements the mid-level computational framework.
- "Testability": A rating from 1 to 10.

Here are the previous high, mid, and low-level theories you've generated:
{previous_ideas}
"""

coherence_check_msg = "You are a critical reviewer evaluating the coherence of a neuroscience theory. You will have {num_rounds_consistency} rounds to review. You do not need to use them all."
coherence_check_prompt = """You are a critical reviewer evaluating the coherence of a neuroscience theory.

This is round {current_round}/{num_rounds_consistency}. 
Evaluate the following neuroscience theory for coherence.

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
1. A low-level insight invalidates or refines the high-level theory.
2. A mid-level framework contradicts or suggests improvements to the high-level theory.
3. The high-level theory is too vague and needs grounding in concrete mechanisms.

Modify any level as needed.

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



novelty_msg = """You are an AI researcher and neuroscientist critically evaluating the novelty of theoretical neuroscience ideas.
Your goal is to determine whether an idea significantly contributes new insights, rather than repeating existing literature.

You will be given a **multi-level neuroscience theory**, consisting of:
1. A **high-level theory** (broad principles and unifying framework).
2. A **mid-level computational model** (specific information processing mechanisms).
3. A **low-level biological mechanism** (synaptic/plasticity-level implementation).

Your job is to:
- Search the neuroscience literature using OpenAlex.
- Identify whether any **existing papers** significantly overlap with any level of the idea.
- Make a decision:
  - If the idea **has a close match**, mark it as **not novel**.
  - If the idea **is not well-explored**, mark it as **novel**.

You will be given {num_rounds_novelty} rounds to decide, but you may stop early if conclusive.

---

NEUROSCIENCE THEORY TO EVALUATE:

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
"""
novelty_prompt = """This is round {current_round}/{num_rounds_novelty}.
You are checking the novelty of a neuroscience theory:

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

The results of the last query are:
{last_query_results}

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, briefly analyze whether the theory is novel or already covered by existing research.
If you find strong overlap, add "Decision made: not novel."  
If no strong overlap is found, add "Decision made: novel."

In <JSON>, return only one field:
- "Query": A search term to find relevant neuroscience papers. 
You must make a query if you have not decided this round. If you've already decided, leave this empty. 
A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.
"""


# format past ideas into a string. based on the level, only show part of the idea.
def format_past_ideas(ideas, level):
    """
    Format past ideas for the given abstraction level.
    """
    if level == "high":
        return "\n".join(
            [f"{i+1}. {idea['High-Level']['Title']}: {idea['High-Level']['Description']}"
             for i, idea in enumerate(ideas)]
        )
    elif level == "mid":
        return "\n".join(
            [f"{i+1}. High-Level: {idea['High-Level']['Title']} - {idea['High-Level']['Description']}\n"
             f"   Mid-Level: {idea['Mid-Level']['Title']} - {idea['Mid-Level']['Description']}"
             for i, idea in enumerate(ideas)]
        )
    elif level == "low":
        return "\n".join(
            [f"{i+1}. High-Level: {idea['High-Level']['Title']} - {idea['High-Level']['Description']}\n"
             f"   Mid-Level: {idea['Mid-Level']['Title']} - {idea['Mid-Level']['Description']}\n"
             f"   Low-Level: {idea['Low-Level']['Title']} - {idea['Low-Level']['Description']}"
             for i, idea in enumerate(ideas)]
        )


# driver function for generating ideas
def generate_ideas(base_dir, client, model, num_ideas, num_rounds_consistency):
    idea_archive = []

    msg_history = []

    for _ in tqdm(range(num_ideas), desc="Generating ideas"):
        try:
            # generate high-level theory
            text, msg_history = get_response_from_llm(
                high_level_prompt.format(previous_ideas=format_past_ideas(idea_archive, "high")), 
                client=client, model=model, system_message=high_level_msg, msg_history=msg_history,
            )
            json_output = extract_json_between_markers(text)
            assert json_output, "Failed to extract JSON for high-level theory"
            high_level = json_output

            # generate mid-level theory
            text, msg_history = get_response_from_llm(
                mid_level_prompt.format(high_level_theory=high_level["Title"], previous_ideas=format_past_ideas(idea_archive, "mid")), 
                client=client, model=model, system_message=mid_level_msg, msg_history=msg_history,
            )
            json_output = extract_json_between_markers(text)
            assert json_output, "Failed to extract JSON for mid-level theory"
            mid_level = json_output

            # generate low-level theory
            text, msg_history = get_response_from_llm(
                low_level_prompt.format(mid_level_model=mid_level["Title"], previous_ideas=format_past_ideas(idea_archive, "low")), 
                client=client, model=model, system_message=low_level_msg, msg_history=msg_history,
            )
            json_output = extract_json_between_markers(text)
            assert json_output, "Failed to extract JSON for low-level theory"
            low_level = json_output


            # coherence check / modification
            for j in range(num_rounds_consistency):
                text, msg_history = get_response_from_llm(
                    coherence_check_prompt.format(
                        current_round=j + 1,
                        num_rounds_consistency=num_rounds_consistency,
                        high_level_json=json.dumps(high_level, indent=2),
                        mid_level_json=json.dumps(mid_level, indent=2),
                        low_level_json=json.dumps(low_level, indent=2),
                    ),
                    client=client,
                    model=model,
                    system_message=coherence_check_msg.format(num_rounds_consistency=num_rounds_consistency),
                    msg_history=msg_history,
                )
                json_output = extract_json_between_markers(text)
                assert json_output, "Failed to extract JSON in coherence check"

                # update the levels
                high_level, mid_level, low_level = json_output["High-Level"], json_output["Mid-Level"], json_output["Low-Level"]

                # done with changes
                if "I am done" in text:
                    print(f"Refinement converged after {j + 1} rounds.")
                    break

            # store past ideas
            idea_archive.append(
                {"High-Level": high_level, "Mid-Level": mid_level, "Low-Level": low_level}
            )

        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    # save generated ideas
    json_path = osp.join(base_dir, "ideas.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idea_archive, f, indent=4)

    print(f"Saved {len(idea_archive)} ideas to {json_path}")

    return idea_archive


# search OpenAlex for neuroscience papers
def search_neuroscience_papers(query, result_limit=10):
    if not query:
        return None

    works = Works().search(query).get(per_page=result_limit)
    if not works:
        return None

    # extract relevant information
    papers = []
    for work in works:
        paper = {
            "title": work.get("title", "Unknown Title"),
            "authors": [a["author"]["display_name"] for a in work.get("authorships", [])],
            "year": work.get("publication_year", "Unknown Year"),
            "venue": work.get("locations", [{}])[0].get("source", {}).get("display_name", "Unknown Venue"),
            "abstract": work.get("abstract", "No abstract available"),
            "citations": work.get("cited_by_count", 0),
        }

        # heuristic: consider papers with >50 citations
        if paper["citations"] > 50:
            papers.append(paper)

    return papers


def check_idea_novelty(base_dir, client, model, num_rounds_novelty):
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
        

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue
            
        print(f"\nChecking novelty of idea {idx}:")

        high_level, mid_level, low_level = idea["High-Level"], idea["Mid-Level"], idea["Low-Level"]
        msg_history = []
        papers_str = ""

        for j in range(num_rounds_novelty):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds_novelty=num_rounds_novelty,
                        high_level_json=json.dumps(high_level, indent=2),
                        mid_level_json=json.dumps(mid_level, indent=2),
                        low_level_json=json.dumps(low_level, indent=2),
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_msg.format(
                        num_rounds_novelty=num_rounds_novelty,
                        high_level_json=json.dumps(high_level, indent=2),
                        mid_level_json=json.dumps(mid_level, indent=2),
                        low_level_json=json.dumps(low_level, indent=2),
                    ),
                    msg_history=msg_history,
                )

                
                if "decision made: novel" in text.lower():
                    print(f"Idea {idx} is novel.")
                    idea["novel"] = True
                    break
                if "decision made: not novel" in text.lower():
                    print(f"Idea {idx} is not novel.")
                    idea["novel"] = False
                    break


                json_output = extract_json_between_markers(text)
                assert json_output, "Failed to extract JSON from LLM output"

                # search for neuroscience papers in OpenAlex
                query = json_output.get("Query", "")
                papers = search_neuroscience_papers(query, result_limit=10)
                if not papers:
                    papers_str = "No relevant papers found."
                    continue

                # format search results
                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        f"{i+1}: {paper['title']} ({paper['year']}) - {', '.join(paper['authors'])}\n"
                        f"Venue: {paper['venue']}\nCitations: {paper['citations']}\n"
                        f"Abstract: {paper['abstract']}\n"
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Failed to check novelty: {e}")
                continue
        
    # save updated ideas
    json_path = os.path.join(base_dir, "ideas.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ideas, f, indent=4)
    
    print(f"Saved novelty-checked ideas to {json_path}")

    return ideas