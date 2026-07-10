import sys
import os

# Add project root folder to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from vault import Vault
from agents.atlas import atlas_route


vault = Vault()

vault.input_data = ["/docs/Santhosh Resume.pdf"]  # Example input data (single document)

atlas_route(vault)

print("Detected Mode:", vault.mode)