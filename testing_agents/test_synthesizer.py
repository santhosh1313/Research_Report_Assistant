import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from vault import Vault
from agents.synthesizer import synthesizer_run


print("===== Synthesizer Test Started =====")


vault = Vault()

vault.mode = "topic"


# manually add Harvester-like facts

vault.log_fact(
    "Artificial Intelligence improves medical diagnosis using deep learning.",
    "research-paper-1"
)

vault.log_fact(
    "AI systems need large datasets and have privacy challenges.",
    "research-paper-2"
)

vault.log_fact(
    "AI can assist doctors but cannot fully replace human experts.",
    "research-paper-3"
)


result = synthesizer_run(vault)


print("\nGenerated Synthesis:\n")

print(result)


print("\n===== Test Completed =====")