import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from vault import Vault
from agents.harvester import harvester_parse_single


print("===== Single Document Test Started =====")


vault = Vault()


vault.input_data = [
    "docs/English_paper.pdf"
]


harvester_parse_single(vault)


print("\nVault Facts:")

for fact in vault.facts:
    print("----------------")
    print(
        fact["text"][:200]
    )

    print(
        "SOURCE:",
        fact["source"]
    )


print(
    "Total pages stored:",
    len(vault.facts)
)