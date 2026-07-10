import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)


for model in genai.list_models():
    print(model.name)
    print(model.supported_generation_methods)
    print("----------------")