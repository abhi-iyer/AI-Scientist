import os
import json
import subprocess

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, extract_json_between_markers_for_math


paper_writing_msg = """
You are an AI researcher and expert scientific writer. Your task is to write a clear and structured scientific white paper
based on a multi-level neuroscience theory. 

This paper should include:
- A well-motivated Abstract summarizing the idea.
- A detailed Introduction covering the background and importance of the problem.
- A rigorous Methods section explaining the theoretical neuroscience framework, models, and biological grounding.
- A structured Discussion section analyzing implications, future directions, and limitations.
- A succinct Conclusion summarizing key insights.

Your response must follow the specified JSON format for each section.
"""
paper_writing_prompts = {
    "Abstract": """Write a structured abstract for a neuroscience paper based on the following multi-level theory:

High-Level Theory:
{high_level_json}

Mid-Level Model:
{mid_level_json}

Low-Level Mechanism:
{low_level_json}

Respond in the following format:
```json
{{
  "Abstract": "<ABSTRACT_TEXT>"
}}
```

Important things to keep in mind during writing the abstract:
- What are we doing and why is it relevant?
- Why is this hard?
- How do we solve it (i.e. our contribution!)?
- How do we verify that we solved it?

Please make sure the abstract reads smoothly. This should be one continuous paragraph with no line breaks.
""",

    "Introduction": """Write an introduction for a neuroscience paper based on the following theory:

High-Level Theory:
{high_level_json}

Mid-Level Model:
{mid_level_json}

Low-Level Mechanism:
{low_level_json}

Explain the motivation behind this research and its significance in neuroscience.
Respond in JSON format:
```json
{{
  "Introduction": "<INTRODUCTION_TEXT>"
}}
```

Important things to keep in mind during writing the introduction:
- This is a longer version of the abstract
- Why is this hard and how do we propose to solve it?
- List contributions or key insights as bulletpoints
""",

    "Methods": """Write a detailed methods section explaining the theoretical neuroscience framework, computational model, and biological mechanisms.

High-Level Theory:
{high_level_json}

Mid-Level Model:
{mid_level_json}

Low-Level Mechanism:
{low_level_json}

Respond in JSON format:
```json
{{
  "Methods": "<METHODS_TEXT>"
}}
```

Important things to keep in mind during writing the methods:
- Mathematical descriptions of the theory if possible. 
- Algorithmic explanations using pseudocode if applicable.
- Fully flesh out as many details as you can. This section should be several paragraphs long, separated by line breaks. 
- Justifications for the theoretical model and biological plausibility using existing literature.
- What are the alternative attempts in literature and why are they insufficient? You can mention exact papers if you remember their names. If experiments describe
long-standing held conclusions in the experimental field, summarize them and explain how they are relevant here.
- Why is this method novel and special? How does it different from other theories already proposed? 

Important formatting instructions:
- Use LaTeX block equations inside `\\begin{{equation}}...\end{{equation}}` instead of inline math (`\$begin:math:text$ ... \\$end:math:text$`). 
- Escape backslashes (`\\`) properly so that the JSON remains valid.
- Do NOT use unescaped double quotes (`"`) inside LaTeX math expressions. If necessary, use single quotes (`'`) instead.
- Use newline characters (`\\n`) for readability** in JSON output.
- Make sure the output is in valid JSON format** following this structure:
""",

    "Discussion": """Write a discussion analyzing the implications, future directions, and limitations of the proposed neuroscience theory.

High-Level Theory:
{high_level_json}

Mid-Level Model:
{mid_level_json}

Low-Level Mechanism:
{low_level_json}

Respond in JSON format:
```json
{{
  "Discussion": "<DISCUSSION_TEXT>"
}}
```

Important things to keep in mind during writing the discussion:
- What are the implications of this theory in neuroscience?
- What are the potential future directions and applications?
- What are the limitations of the theory and possible ways to address them?
- How does this theory compare to existing theories in the field?
""",

    "Conclusion": """Write a succinct conclusion summarizing the key insights from this neuroscience paper.

High-Level Theory:
{high_level_json}

Mid-Level Model:
{mid_level_json}

Low-Level Mechanism:
{low_level_json}

Respond in JSON format:
```json
{{
  "Conclusion": "<CONCLUSION_TEXT>"
}}
```

Important things to keep in mind during writing the conclusion:
- Summarize the key insights and contributions of the paper.
- Restate the importance of the research and its potential impact in neuroscience.
- Suggest possible future research directions based on the findings.
"""
}


# LaTeX template
latex_template = r"""
\documentclass{{article}}
\usepackage{{amsmath, amssymb}}
\usepackage{{algorithm, algorithmicx, algpseudocode}}

\newcommand{{\normalsizeauthor}}{{\normalsize}}

\title{{{title}}}
\author{{\normalsizeauthor AI Neuroscientist}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\section{{Introduction}}
{introduction}

\section{{Methods}}
{methods}

\section{{Discussion}}
{discussion}

\section{{Conclusion}}
{conclusion}

\end{{document}}
"""


# driver function for generating the paper
def generate_paper(base_dir, client, model):
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
        fname = idea["High-Level"].get("Title", f"idea{idx}")

        high_level, mid_level, low_level = (
            json.dumps(idea["High-Level"], indent=2),
            json.dumps(idea["Mid-Level"], indent=2),
            json.dumps(idea["Low-Level"], indent=2),
        )
    
        paper_content = {}
        msg_history = []
    
        for section, prompt in paper_writing_prompts.items():
            text, msg_history = get_response_from_llm(
                prompt.format(
                    high_level_json=high_level,
                    mid_level_json=mid_level,
                    low_level_json=low_level,
                ),
                client=client,
                model=model,
                system_message=paper_writing_msg,
                msg_history=msg_history,
            )

            extracted = extract_json_between_markers(text)

            if extracted:
                paper_content[section] = extracted.get(section, "")
            else:
                # try brute force
                
                paper_content[section] = extract_json_between_markers_for_math(text).get(section, "")

        latex_doc = latex_template.format(
            title=idea["High-Level"].get("Title", "Untitled"),
            abstract=paper_content["Abstract"],
            introduction=paper_content["Introduction"],
            methods=paper_content["Methods"],
            discussion=paper_content["Discussion"],
            conclusion=paper_content["Conclusion"],
        )
            

        latex_dir = os.path.join(base_dir, "latex")
        os.makedirs(latex_dir, exist_ok=True)
        tex_file = os.path.join(latex_dir, fname + ".tex")

        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(latex_doc)
        
        compile_latex(latex_dir, fname + ".tex")
    
        print(f"Generated paper saved at: {os.path.join(latex_dir, f'{fname}.pdf')}")


def compile_latex(cwd, fname):
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", fname], cwd=cwd, check=True)
        print("Successfully compiled LaTeX document.")
    except subprocess.CalledProcessError:
        print("Error: Failed to compile LaTeX document.")