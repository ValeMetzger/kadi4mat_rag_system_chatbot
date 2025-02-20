def clean_response(response: str) -> str:
    """Clean up model response."""
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1]
    
    response = response.replace("<s>", "").replace("</s>", "")
    response = response.replace("[INST]", "").strip()
    response = " ".join(response.split())
    
    return response