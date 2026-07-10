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
 
    response = llm.invoke(prompt)
    vault.subtasks = response.content.split("\n")
    print(f"[Pathfinder] Plan: {vault.subtasks}")
    return vault.subtasks
