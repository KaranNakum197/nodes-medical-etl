"""
CrewAI setup – agent definitions for the medical ETL pipeline.

Agents
------
extractor   – Runs Qwen2-VL via the backend API to pull structured data.
validator   – Validates and cleans the extracted JSON against medical rules.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Process
from langchain_openai import ChatOpenAI

from tasks import build_tasks

# ---------------------------------------------------------------------------
# Language model (used as the "brain" for orchestration decisions)
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

extractor_agent = Agent(
    role="Medical Document Extractor",
    goal=(
        "Convert raw medical documents into structured JSON by invoking the "
        "VLM inference API and returning complete, accurate records."
    ),
    backstory=(
        "You are an expert medical data analyst with deep knowledge of clinical "
        "documentation. You use the Qwen2-VL model to read scanned reports and "
        "extract key fields with high precision."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

validator_agent = Agent(
    role="Medical Data Validator",
    goal=(
        "Validate and normalise extracted medical data. Flag missing fields, "
        "date format errors, and implausible values."
    ),
    backstory=(
        "You are a senior clinical data curator responsible for ensuring that "
        "every record entering the database is complete, consistent, and "
        "compliant with standard medical coding conventions."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Crew assembly
# ---------------------------------------------------------------------------

def build_crew(document_path: str) -> Crew:
    """
    Assemble and return the ETL crew for a given document.

    Parameters
    ----------
    document_path:
        Absolute path to the medical document to process.
    """
    tasks = build_tasks(
        extractor=extractor_agent,
        validator=validator_agent,
        document_path=document_path,
    )

    return Crew(
        agents=[extractor_agent, validator_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    crew = build_crew(path)
    result = crew.kickoff()
    print("\n=== Final Result ===")
    print(result)
