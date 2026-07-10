import sys
import os


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from vault import Vault
from agents.harvester import harvester_embed_multi


print("===== Harvester Multi PDF Test =====")


vault = Vault()


vault.input_data = [
    "docs/appreciation.pdf",
    "docs/satisfaction.pdf"
]


harvester_embed_multi(vault)


print(
    "Vector store:",
    vault.vector_store
)


print("Multi PDF Test Completed")