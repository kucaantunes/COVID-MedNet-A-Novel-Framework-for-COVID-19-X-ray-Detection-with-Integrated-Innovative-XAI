def simulate_vlm_response(prompt):
    from text_generator import generate_text
    response = generate_text(prompt, is_vlm=True)
    return response