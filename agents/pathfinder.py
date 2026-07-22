# agents/pathfinder.py
from dotenv import load_dotenv
import os

load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


class PathfinderError(Exception):
    """Raised when planning fails or produces no usable subtasks."""
    pass


def pathfinder_plan(vault):
    if vault.mode == "topic":
        prompt = f"""Break this research topic into 3-5 focused sub-questions
        that together would produce a well-rounded report.
        Topic: {vault.input_data}
        Return only a numbered list."""
    else:
        prompt = """List 4-5 comparison dimensions relevant to analyzing
        academic papers (e.g., methodology, dataset, key findings, limitations).
        Return only a numbered list."""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise PathfinderError(f"Groq LLM call failed during planning: {e}") from e

    content = getattr(response, "content", None)
    if not content or not content.strip():
        raise PathfinderError("Pathfinder LLM returned an empty plan")

    subtasks = [line.strip() for line in content.split("\n") if line.strip()]
    if not subtasks:
        raise PathfinderError("Pathfinder plan parsed to zero usable subtasks")

    vault.subtasks = subtasks
    print(f"[Pathfinder] Plan: {vault.subtasks}")
    return vault.subtasks