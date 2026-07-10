# agents/scribe.py

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# =================================================
# MODE 1: TOPIC RESEARCH REPORT
# =================================================

TOPIC_PROMPT = """
You are a professional research analyst.

Create a research report from information collected from multiple web sources.

Structure:

# Title

## Executive Summary
Explain:
- research topic
- important findings
- final conclusion

## Introduction
Explain:
- background
- importance of the topic

## Key Research Themes
Group findings into meaningful themes.

For each theme:
- explain findings
- provide evidence
- use inline citations like [1], [2]

## Different Perspectives and Contradictions
Analyze:
- where sources agree
- where findings conflict
- reasons behind differences

## Current Trends / Real World Applications
Explain:
- current industry usage
- latest developments

## Conclusion
Include:
- final summary
- future outlook

## References

Rules for references:
- Use numbered references only.
- Example: [1] https://source-url.com
- Every inline citation must have a matching reference.
- Do not create fake references.
- Do not duplicate the same URL.
- Merge repeated sources.

Writing Rules:
- Length: 800-1500 words.
- Maintain academic tone.
- Remove duplicate sentences.
- Remove repeated paragraphs.
- Avoid mentioning the same finding multiple times.
- Proofread before returning.

Synthesis:

{synthesis}
"""


# =================================================
# MODE 2: SINGLE PAPER SUMMARY
# =================================================

SINGLE_DOC_PROMPT = """
You are an academic paper analysis assistant.

Create a structured summary of ONE research paper.

Do NOT:
- compare with other papers
- create comparison tables
- create agreement/contradiction sections

Structure:

# Paper Title

## Overview
Explain:
- research problem
- motivation
- objective

## Methodology Explained
Describe:
- proposed method
- models/algorithms used
- workflow

## Key Contributions
Mention:
- innovations
- improvements

## Experimental Setup
Include if available:
- dataset
- metrics
- evaluation method

## Results and Findings
Explain:
- results achieved
- observations

## Internal Consistency Analysis
Analyze:
- whether claims match results
- whether conclusions are supported

## Limitations
Mention:
- weaknesses
- assumptions
- missing evaluations

## Conclusion
Summarize:
- importance
- future improvements

Rules:
- Explain clearly.
- Do not add unsupported information.
- Remove duplicate content.

Paper Analysis:

{synthesis}
"""


# =================================================
# MODE 3: MULTI PAPER LITERATURE REVIEW
# =================================================

MULTI_DOC_PROMPT = """
You are a research literature review expert.

Create a comparative review of multiple papers.

Structure:

# Title

## Introduction
Explain:
- research area
- importance
- papers compared

## Individual Paper Summaries

For each paper include:
- objective
- methodology
- findings
- limitations

## Comparative Analysis Table

Create table:

| Paper | Method | Dataset | Results | Advantages | Limitations |

## Common Findings
Explain:
- agreements
- shared approaches

## Contradictions / Differences
Explain:
- different methods
- conflicting results

## Research Gaps
Identify:
- missing experiments
- open problems
- future research scope

## Final Evaluation
Discuss:
- strongest approach
- practical usefulness

## Conclusion
Provide final synthesized takeaway.

Rules:
- Compare instead of only summarizing.
- Preserve citations.
- Avoid unsupported claims.
- Remove duplicate content.

Research Synthesis:

{synthesis}
"""


# =================================================
# SCRIBE AGENT
# =================================================

def scribe_write(vault):

    if vault.mode == "topic":
        prompt = TOPIC_PROMPT.format(
            synthesis=vault.synthesis
        )

    elif vault.mode == "single_doc":
        prompt = SINGLE_DOC_PROMPT.format(
            synthesis=vault.synthesis
        )

    elif vault.mode == "multi_doc":
        prompt = MULTI_DOC_PROMPT.format(
            synthesis=vault.synthesis
        )

    else:
        raise ValueError(
            f"Unknown mode: {vault.mode}"
        )

    response = llm.invoke(prompt)

    vault.final_report = response.content

    print(
        f"[Scribe] {vault.mode} report drafted"
    )

    return vault.final_report