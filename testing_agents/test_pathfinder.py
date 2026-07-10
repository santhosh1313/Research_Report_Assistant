import sys
import os

# Add project root folder to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from vault import Vault
from agents.pathfinder import pathfinder_plan


print("===== Pathfinder Agent Test Started =====")


# Create Vault object (shared memory)
vault = Vault()


# Give sample topic
vault.input_data = "Impact of Artificial Intelligence in Healthcare"


# Since testing Pathfinder alone,
# set mode manually (without Atlas)
vault.mode = "topic"


print("\nInput Topic:")
print(vault.input_data)


# Run Pathfinder agent
pathfinder_plan(vault)


print("\nSubtasks Stored in Vault:")

for task in vault.subtasks:
    print(task)


print("\n===== Pathfinder Test Completed =====")