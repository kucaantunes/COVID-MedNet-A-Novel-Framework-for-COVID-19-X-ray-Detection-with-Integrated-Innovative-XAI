def simulate_llm_response(prompt):
    from text_generator import generate_text
    response = generate_text(prompt, is_vlm=False)
    return response

def chatmedbot_response(question, prediction, confidence, vital_signs):
    from text_generator import generate_text
    
    base_prompt = (
        f"You are an advanced AI assistant capable of discussing any topic, similar to ChatGPT, with expertise in medical analysis. "
        f"Answer the question: '{question}'. If relevant, integrate details from a chest X-ray analysis: diagnosis={prediction}, "
        f"confidence={confidence:.2f}, vital signs (Heart Rate={vital_signs.get('HR', 'N/A')} bpm, "
        f"Respiratory Rate={vital_signs.get('RR', 'N/A')} breaths/min, Oxygen Saturation={vital_signs.get('SpO2', 'N/A')}%, "
        f"Blood Pressure={vital_signs.get('BP', 'N/A')} mmHg, Temperature={vital_signs.get('Temp', 'N/A')}C), "
        f"and HeatmapFocusXAI insights (red: most used, orange: medium used, yellow: least used by the algorithm). "
        f"Provide a detailed, conversational response, explaining X-ray features, vital signs correlation, and XAI focus areas."
    )
    response = generate_text(base_prompt, is_vlm=False)
    return response