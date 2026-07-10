import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from vault import Vault
from agents.harvester import harvester_web_search


print("===== Harvester Web Test =====")


vault = Vault()


# Simulating Pathfinder output
vault.subtasks = [
    "Uses of AI in education",
    "Benefits of artificial intelligence"
]


harvester_web_search(vault)


print("\nVault Facts:")

for fact in vault.facts:
    print("\nFACT:")
    print(fact["text"][:300])
    print("SOURCE:", fact["source"])


print("\nTotal Facts:", len(vault.facts))