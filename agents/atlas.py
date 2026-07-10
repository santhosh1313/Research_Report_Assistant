# agents/atlas.py
def detect_mode(input_data):
    """
    input_data: either a string (topic) or a list of file paths (documents)
    """
    if isinstance(input_data, str):
        return "topic"
    elif isinstance(input_data, list) and len(input_data) == 1:
        return "single_doc"
    elif isinstance(input_data, list) and len(input_data) > 1:
        return "multi_doc"
    else:
        raise ValueError("Unrecognized input type")
 
def atlas_route(vault):
    vault.mode = detect_mode(vault.input_data)
    print(f"[Atlas] Mode detected: {vault.mode}")
    return vault.mode
