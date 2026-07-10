import sys
import os


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from vault import Vault
from agents.scribe import scribe_write


print("===== Scribe Test Started =====")


# Create shared memory
vault = Vault()


# Fake output from Synthesizer Agent
vault.synthesis = """
Artificial Intelligence is transforming healthcare.

Key Findings:
- AI improves disease diagnosis.
- Deep learning helps analyze medical images.
- AI assistants support doctors.

Challenges:
- Data privacy issues.
- Model bias.
- Need for human supervision.

Sources:
paper1.pdf
paper2.pdf
"""


# Run Scribe
report = scribe_write(vault)


print("\nGenerated Final Report:\n")


print(report)


print("\n===== Scribe Test Completed =====")