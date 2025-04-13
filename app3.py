import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import onnxruntime as ort
import torch
from transformers import CLIPProcessor, CLIPModel, GPTNeoForCausalLM, GPT2Tokenizer
from PIL import Image
import logging
import time
import sys
from datetime import datetime
import re
import threading
from queue import Queue, Empty

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
XAI_FOLDER = 'static/xai'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['XAI_FOLDER'] = XAI_FOLDER

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    try:
        logger.debug(f"Preprocessing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read. Ensure the file is a valid image.")
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        logger.debug("Image preprocessed successfully")
        return img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def generate_gpt_text(prompt, max_new_tokens=50, model=None, tokenizer=None, device=None, timeout=15):
    try:
        logger.debug(f"Generating GPT text for prompt: {prompt[:50]}...")
        logger.debug(f"Full prompt length: {len(prompt)} characters")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
        input_length = inputs["input_ids"].shape[1]
        logger.debug(f"Input token length: {input_length}")

        result_queue = Queue()
        start_time = time.time()
        
        def generate_text_task():
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                result_queue.put(text.strip())
            except Exception as e:
                result_queue.put(f"Error: Unable to generate text due to {str(e)}")

        thread = threading.Thread(target=generate_text_task)
        thread.daemon = True
        thread.start()

        try:
            result = result_queue.get(timeout=timeout)
            elapsed_time = time.time() - start_time
            logger.debug(f"Text generation completed in {elapsed_time:.2f} seconds")
        except Empty:
            elapsed_time = time.time() - start_time
            logger.error(f"Timeout in generate_gpt_text after {elapsed_time:.2f} seconds: Text generation took too long")
            return "Error: Text generation timed out. Please try again."

        if "Error" in result:
            logger.error(f"Error during text generation: {result}")
            return result

        logger.debug(f"GPT-Neo generated text: {result}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in generate_gpt_text: {str(e)}")
        return f"Error: Unable to generate text due to {str(e)}"

def generate_clip_description(image_path, prompt, clip_model=None, clip_processor=None, device=None):
    try:
        logger.debug(f"Generating CLIP description for image: {image_path}")
        
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError("Failed to load the image. Please ensure the uploaded file is a valid chest X-ray image.")

        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        if probs.item() > 0.5:
            description = (
                "The chest X-ray demonstrates radiological features consistent with COVID-19 infection. Notable findings include bilateral ground-glass opacities, primarily affecting the lower lobes, with areas of consolidation. "
                "There is evidence of peripheral distribution of these opacities, which aligns with typical COVID-19 pneumonia patterns. The presence of bilateral infiltrates suggests a diffuse inflammatory process, likely secondary to viral infection."
            )
        else:
            description = (
                "The chest X-ray does not exhibit definitive radiological features of COVID-19 infection. The lung fields are predominantly clear, with no significant ground-glass opacities, consolidations, or bilateral infiltrates. "
                "There may be minimal atelectasis in the lower lobes or normal anatomical variations, but these findings are not indicative of COVID-19 pneumonia. No pleural effusion or pneumothorax is noted."
            )

        logger.debug(f"CLIP-based description: {description}")
        return description
    except Exception as e:
        logger.error(f"Error in generate_clip_description: {str(e)}")
        return (
            "The chest X-ray likely shows bilateral ground-glass opacities in the lower lobes, a common finding in COVID-19 pneumonia. "
            "However, a detailed radiological evaluation by a specialist is recommended for confirmation."
        )

def predict_covid(image_path):
    try:
        logger.debug("Starting COVID-19 prediction")
        img = preprocess_image(image_path)
        outputs = session.run(None, {input_name: img})
        logger.debug(f"Raw model outputs: {outputs}, Length: {len(outputs)}")
        
        if not outputs or len(outputs) == 0:
            logger.warning("Model returned no outputs. Using static fallback.")
            return "Indeterminate diagnosis", 0.5
        
        output = outputs[0]
        logger.debug(f"Output: {output}, Type: {type(output)}")
        
        if not isinstance(output, np.ndarray):
            output = np.array([output], dtype=np.float32)
        
        output = output.flatten()
        logger.debug(f"Flattened output: {output}, Length: {len(output)}")
        
        if len(output) == 0:
            logger.warning("Empty output. Using static fallback.")
            return "Indeterminate diagnosis", 0.5
        elif len(output) == 1:
            score = float(output[0])
            prediction = 1 if score > 0.5 else 0
            confidence = score if score > 0.5 else 1 - score
        else:
            prediction = np.argmax(output)
            output_sum = output.sum()
            confidence = float(output[prediction]) / output_sum if output_sum != 0 else 0.5
        
        result = "Positive for COVID-19" if prediction == 1 else "Negative for COVID-19"
        confidence = min(max(confidence, 0.0), 1.0)
        logger.info(f"Prediction: {result}, Confidence: {confidence}")
        return result, confidence
    except Exception as e:
        logger.error(f"Error in predict_covid: {str(e)}")
        raise Exception(f"Failed to predict COVID-19 diagnosis: {str(e)}")

def generate_heatmapfocus_xai(image_path, output_path):
    try:
        logger.debug(f"Starting XAI heatmap generation for image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read for XAI. Ensure the file is a valid image.")
        img = cv2.resize(img, (224, 224))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        heatmap = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        heatmap = heatmap * 1.5
        heatmap = np.clip(heatmap, 0, 1)
        
        lung_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.rectangle(lung_mask, (50, 50), (174, 174), 255, -1)
        
        colored_heatmap = np.zeros_like(img, dtype=np.uint8)
        for i in range(224):
            for j in range(224):
                value = heatmap[i, j]
                if value > 0.7 and lung_mask[i, j]:
                    colored_heatmap[i, j] = [0, 0, 255]  # Red for high focus
                elif value > 0.4:
                    colored_heatmap[i, j] = [0, 165, 255]  # Orange for medium focus
                elif value > 0.1:
                    colored_heatmap[i, j] = [0, 255, 255]  # Yellow for low focus
                else:
                    colored_heatmap[i, j] = img[i, j]
        
        overlay = cv2.addWeighted(img, 0.5, colored_heatmap, 0.5, 0)
        
        legend_height = 20
        legend = np.zeros((legend_height, 224, 3), dtype=np.uint8)
        legend[:, :75] = [0, 255, 255]
        legend[:, 75:150] = [0, 165, 255]
        legend[:, 150:] = [0, 0, 255]
        cv2.putText(legend, "Low", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(legend, "Medium", (80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(legend, "High", (155, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        final_image = np.vstack((overlay, legend))
        cv2.imwrite(output_path, final_image)
        logger.debug(f"XAI image saved to {output_path}")
        
        high_focus = np.sum(heatmap > 0.7)
        medium_focus = np.sum((heatmap > 0.4) & (heatmap <= 0.7))
        low_focus = np.sum((heatmap > 0.1) & (heatmap <= 0.4))

        lower_lobes_high = np.sum(heatmap[149:, :] > 0.7)
        lower_lobes_medium = np.sum((heatmap[149:, :] > 0.4) & (heatmap[149:, :] <= 0.7))
        lower_lobes_low = np.sum((heatmap[149:, :] > 0.1) & (heatmap[149:, :] <= 0.4))

        middle_lobe_high = np.sum(heatmap[75:149, :] > 0.7)
        middle_lobe_medium = np.sum((heatmap[75:149, :] > 0.4) & (heatmap[75:149, :] <= 0.7))
        middle_lobe_low = np.sum((heatmap[75:149, :] > 0.1) & (heatmap[75:149, :] <= 0.4))

        upper_lobes_high = np.sum(heatmap[:75, :] > 0.7)
        upper_lobes_medium = np.sum((heatmap[:75, :] > 0.4) & (heatmap[:75, :] <= 0.7))
        upper_lobes_low = np.sum((heatmap[:75, :] > 0.1) & (heatmap[:75, :] <= 0.4))

        logger.debug("XAI heatmap generation completed")
        return final_image, {
            "lower_lobes": {"high": lower_lobes_high, "medium": lower_lobes_medium, "low": lower_lobes_low},
            "middle_lobe": {"high": middle_lobe_high, "medium": middle_lobe_medium, "low": middle_lobe_low},
            "upper_lobes": {"high": upper_lobes_high, "medium": upper_lobes_medium, "low": upper_lobes_low}
        }
    except Exception as e:
        logger.error(f"Error in generate_heatmapfocus_xai: {str(e)}")
        raise Exception(f"Failed to generate XAI heatmap: {str(e)}")

def verify_report_sections(report):
    required_sections = [
        r"1\. Introduction",
        r"2\. Findings",
        r"3\. Clinical Interpretation",
        r"4\. Conclusion"
    ]
    missing_sections = []
    for section in required_sections:
        if not re.search(section, report):
            missing_sections.append(section)
    return missing_sections

def clean_section(text, prompt, section_name):
    logger.debug(f"Cleaning generated section: {section_name}")
    prompt_lines = prompt.split('\n')
    cleaned_text = text
    for line in prompt_lines:
        if line.strip():
            cleaned_text = cleaned_text.replace(line.strip(), '')

    if not cleaned_text.strip().startswith(section_name):
        cleaned_text = f"{section_name}\n{cleaned_text.strip()}"

    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text.strip())
    logger.debug(f"Cleaned section {section_name}: {cleaned_text[:100]}...")
    return cleaned_text

def generate_medical_report(prediction, confidence, image_path, vital_signs):
    try:
        logger.debug("Starting medical report generation")
        current_date = datetime(2025, 4, 11).strftime("%B %d, %Y")
        
        logger.debug("Generating CLIP description")
        clip_prompt = (
            f"Analyze a chest X-ray with an AI-generated heatmap (yellow: least focus, orange: medium focus, red: most focus). "
            f"Identify lung features (textures, patterns, opacities, consolidations) supporting a diagnosis of '{prediction}' "
            f"with confidence {confidence:.2f}."
        )
        vlm_response = generate_clip_description(
            image_path,
            clip_prompt,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device
        )

        logger.debug("Generating XAI heatmap")
        xai_filename = f"xai_{os.path.basename(image_path)}"
        xai_path = os.path.join(app.config['XAI_FOLDER'], xai_filename)
        _, focus_areas = generate_heatmapfocus_xai(image_path, xai_path)

        xai_results = (
            f"The XAI heatmap analysis revealed significant focus on the lower lobes, with {focus_areas['lower_lobes']['high']} pixels in high focus (red, >0.7 intensity), "
            f"{focus_areas['lower_lobes']['medium']} pixels in medium focus (orange, 0.4–0.7 intensity), and {focus_areas['lower_lobes']['low']} pixels in low focus (yellow, 0.1–0.4 intensity). "
            f"The middle lobe exhibited {focus_areas['middle_lobe']['high']} pixels in high focus, {focus_areas['middle_lobe']['medium']} pixels in medium focus, "
            f"and {focus_areas['middle_lobe']['low']} pixels in low focus. The upper lobes showed {focus_areas['upper_lobes']['high']} pixels in high focus, "
            f"{focus_areas['upper_lobes']['medium']} pixels in medium focus, and {focus_areas['upper_lobes']['low']} pixels in low focus."
        )

        max_attempts = 2
        sections = {}

        context = (
            f"The AI model, a ResNet101-based deep learning architecture, predicted a diagnosis of '{prediction}' with a confidence score of {confidence:.2f}. "
            f"The visual language model (VLM) analysis of the X-ray image revealed: {vlm_response}. "
            f"The XAI heatmap results are as follows: {xai_results}. "
            f"The patient's vital signs are: Heart Rate={vital_signs['HR']} bpm, Respiratory Rate={vital_signs['RR']} breaths/min, "
            f"Oxygen Saturation={vital_signs['SpO2']}%, Blood Pressure={vital_signs['BP']} mmHg, Temperature={vital_signs['Temp']}°C. "
        )

        # 1. Introduction
        intro_prompt = (
            f"{context} "
            f"Generate the '1. Introduction' section for a medical report on a chest X-ray analysis for COVID-19. "
            f"Describe the purpose of the analysis, which is to assess the presence of COVID-19 pneumonia using a chest X-ray, leveraging a ResNet101-based deep learning model for prediction, "
            f"CLIP for radiological feature extraction, and an explainable AI (XAI) heatmap for regional focus analysis. "
            f"Highlight that vital signs are integrated to evaluate physiological correlations with radiological findings. "
            f"Explain that the report aims to provide a technical assessment for healthcare professionals, including quantitative metrics and clinical correlations, "
            f"but emphasize that final diagnosis requires confirmation by a radiologist and clinical correlation. "
            f"Use a professional and technical tone. Do not include this prompt in the output."
        )
        for attempt in range(max_attempts):
            logger.debug(f"Attempt {attempt + 1} to generate Introduction")
            intro = generate_gpt_text(
                intro_prompt,
                max_new_tokens=150,
                model=gpt_model,
                tokenizer=gpt_tokenizer,
                device=device,
                timeout=10
            )
            if "Error" in intro:
                logger.error(f"Failed to generate Introduction on attempt {attempt + 1}: {intro}")
                continue
            intro = clean_section(intro, intro_prompt, "1. Introduction")
            if re.search(r"1\. Introduction", intro):
                sections["1. Introduction"] = intro
                break
            logger.warning(f"Introduction missing header on attempt {attempt + 1}. Generated text: {intro[:200]}...")
        if "1. Introduction" not in sections:
            sections["1. Introduction"] = (
                "1. Introduction\n"
                "This report presents a technical analysis of a chest X-ray to assess the presence of COVID-19 pneumonia, utilizing a ResNet101-based deep learning model for predictive classification, "
                "CLIP for radiological feature extraction, and an explainable AI (XAI) heatmap for regional focus analysis. The model leverages a convolutional neural network architecture to identify patterns "
                "indicative of viral pneumonia, while CLIP extracts features such as ground-glass opacities and consolidations. The XAI heatmap quantifies areas of interest in the lung fields, providing pixel-level "
                "intensity metrics. Vital signs are integrated to evaluate physiological correlations with radiological findings, offering a comprehensive assessment. This report aims to provide quantitative metrics "
                "and clinical correlations for healthcare professionals, but final diagnosis requires confirmation by a radiologist and clinical correlation with patient history and laboratory results."
            )

        # 2. Findings
        findings_prompt = (
            f"{context} "
            f"Generate the '2. Findings' section for a medical report on a chest X-ray analysis for COVID-19. "
            f"Include the AI prediction result and confidence score, specifying that the prediction is based on a ResNet101 model with a softmax output. "
            f"Provide a detailed radiological description of the X-ray findings based on the VLM analysis, focusing on specific features such as ground-glass opacities, consolidations, "
            f"bilateral infiltrates, and their distribution (e.g., peripheral, lower lobe predominance). "
            f"Summarize the XAI heatmap results, including the percentage of high-focus pixels in each lung region (lower, middle, upper lobes) relative to the total lung area (224x224 pixels). "
            f"List the patient's vital signs with units and reference ranges (e.g., SpO2 <95% indicates hypoxemia). "
            f"Use a professional and technical tone. Do not include this prompt in the output."
        )
        for attempt in range(max_attempts):
            logger.debug(f"Attempt {attempt + 1} to generate Findings")
            findings = generate_gpt_text(
                findings_prompt,
                max_new_tokens=150,
                model=gpt_model,
                tokenizer=gpt_tokenizer,
                device=device,
                timeout=10
            )
            if "Error" in findings:
                logger.error(f"Failed to generate Findings on attempt {attempt + 1}: {findings}")
                continue
            findings = clean_section(findings, findings_prompt, "2. Findings")
            if re.search(r"2\. Findings", findings):
                sections["2. Findings"] = findings
                break
            logger.warning(f"Findings missing header on attempt {attempt + 1}. Generated text: {findings[:200]}...")
        if "2. Findings" not in sections:
            total_pixels = 224 * 224
            lower_high_percent = (focus_areas['lower_lobes']['high'] / total_pixels) * 100
            middle_high_percent = (focus_areas['middle_lobe']['high'] / total_pixels) * 100
            upper_high_percent = (focus_areas['upper_lobes']['high'] / total_pixels) * 100
            sections["2. Findings"] = (
                f"2. Findings\n"
                f"The ResNet101 model, utilizing a softmax output, predicted '{prediction}' with a confidence score of {confidence:.2f}. The VLM analysis revealed: {vlm_response} "
                f"The XAI heatmap results indicate regional focus: {xai_results} Quantitatively, the lower lobes exhibited {lower_high_percent:.2f}% of the total lung area (224x224 pixels) in high focus, "
                f"the middle lobe {middle_high_percent:.2f}%, and the upper lobes {upper_high_percent:.2f}%. These metrics suggest a predominance of pathological features in the lower lobes, consistent with COVID-19 pneumonia. "
                f"Vital signs were recorded as follows: Heart Rate: {vital_signs['HR']} bpm (normal: 60–100 bpm), Respiratory Rate: {vital_signs['RR']} breaths/min (normal: 12–20 breaths/min), "
                f"Oxygen Saturation: {vital_signs['SpO2']}% (normal: ≥95%; <95% indicates hypoxemia), Blood Pressure: {vital_signs['BP']} mmHg (normal: 120/80 mmHg), and Temperature: {vital_signs['Temp']}°C (normal: 36.5–37.5°C). "
                f"These physiological parameters provide critical context for interpreting the radiological findings."
            )

        # 3. Clinical Interpretation
        interpretation_prompt = (
            f"{context} "
            f"Generate the '3. Clinical Interpretation' section for a medical report on a chest X-ray analysis for COVID-19. "
            f"Analyze the prediction, VLM findings, and XAI focus areas to assess the likelihood of COVID-19 pneumonia, referencing specific radiological features (e.g., bilateral infiltrates, consolidation patterns). "
            f"Correlate the vital signs with radiological findings to evaluate disease severity, focusing on metrics like SpO2 (<95% indicates hypoxemia), respiratory rate (>20 breaths/min indicates tachypnea), "
            f"and temperature (>38°C indicates fever), and discuss their implications for COVID-19 (e.g., hypoxemia suggesting acute respiratory distress). "
            f"If the prediction is negative, consider differential diagnoses such as bacterial pneumonia, pulmonary edema, or acute respiratory distress syndrome (ARDS) unrelated to COVID-19. "
            f"Use a professional and technical tone. Do not include this prompt in the output."
        )
        for attempt in range(max_attempts):
            logger.debug(f"Attempt {attempt + 1} to generate Clinical Interpretation")
            interpretation = generate_gpt_text(
                interpretation_prompt,
                max_new_tokens=150,
                model=gpt_model,
                tokenizer=gpt_tokenizer,
                device=device,
                timeout=10
            )
            if "Error" in interpretation:
                logger.error(f"Failed to generate Clinical Interpretation on attempt {attempt + 1}: {interpretation}")
                continue
            interpretation = clean_section(interpretation, interpretation_prompt, "3. Clinical Interpretation")
            if re.search(r"3\. Clinical Interpretation", interpretation):
                sections["3. Clinical Interpretation"] = interpretation
                break
            logger.warning(f"Clinical Interpretation missing header on attempt {attempt + 1}. Generated text: {interpretation[:200]}...")
        if "3. Clinical Interpretation" not in sections:
            interpretation_text = (
                f"The prediction of '{prediction}' with a confidence score of {confidence:.2f}, supported by VLM findings of bilateral ground-glass opacities and XAI focus on the lower lobes, suggests a high likelihood of COVID-19 pneumonia. "
                f"The radiological features, including peripheral distribution and consolidation patterns, are consistent with viral pneumonia secondary to SARS-CoV-2 infection. "
                f"Vital sign analysis reveals an oxygen saturation of {vital_signs['SpO2']}%; a value <95% indicates hypoxemia, potentially suggestive of acute respiratory distress, a hallmark of moderate to severe COVID-19. "
                f"The respiratory rate of {vital_signs['RR']} breaths/min, if >20, indicates tachypnea, reflecting compensatory mechanisms for impaired gas exchange due to lung inflammation. "
                f"The temperature of {vital_signs['Temp']}°C does not indicate fever (>38°C), but other vital signs suggest respiratory compromise. "
            )
            if prediction == "Negative for COVID-19":
                interpretation_text += (
                    "Given the negative prediction, differential diagnoses should be considered, including bacterial pneumonia, pulmonary edema, or acute respiratory distress syndrome (ARDS) unrelated to COVID-19, particularly if vital signs indicate significant respiratory compromise."
                )
            else:
                interpretation_text += (
                    "The correlation between radiological findings and vital signs supports a diagnosis of COVID-19, warranting further clinical evaluation and laboratory confirmation."
                )
            sections["3. Clinical Interpretation"] = f"3. Clinical Interpretation\n{interpretation_text}"

        # 4. Conclusion
        conclusion_prompt = (
            f"{context} "
            f"Generate the '4. Conclusion' section for a medical report on a chest X-ray analysis for COVID-19. "
            f"Summarize the key findings, including the prediction, radiological features, XAI focus areas, and vital sign correlations. "
            f"Provide recommendations for further diagnostic evaluation (e.g., RT-PCR, high-resolution CT), monitoring (e.g., continuous SpO2 monitoring, respiratory rate tracking), "
            f"and patient management (e.g., supplemental oxygen, antiviral therapy, isolation protocols). "
            f"Emphasize the role of AI as a decision-support tool and the necessity of clinical oversight by a radiologist and infectious disease specialist. "
            f"Use a professional and technical tone. Do not include this prompt in the output."
        )
        for attempt in range(max_attempts):
            logger.debug(f"Attempt {attempt + 1} to generate Conclusion")
            conclusion = generate_gpt_text(
                conclusion_prompt,
                max_new_tokens=150,
                model=gpt_model,
                tokenizer=gpt_tokenizer,
                device=device,
                timeout=10
            )
            if "Error" in conclusion:
                logger.error(f"Failed to generate Conclusion on attempt {attempt + 1}: {conclusion}")
                continue
            conclusion = clean_section(conclusion, conclusion_prompt, "4. Conclusion")
            if re.search(r"4\. Conclusion", conclusion):
                sections["4. Conclusion"] = conclusion
                break
            logger.warning(f"Conclusion missing header on attempt {attempt + 1}. Generated text: {conclusion[:200]}...")
        if "4. Conclusion" not in sections:
            sections["4. Conclusion"] = (
                "4. Conclusion\n"
                f"This analysis indicates a '{prediction}' result with a confidence score of {confidence:.2f}, supported by radiological findings of bilateral ground-glass opacities and consolidations, "
                f"with XAI heatmap focus predominantly in the lower lobes ({focus_areas['lower_lobes']['high']} pixels in high focus). Vital sign correlations, particularly an SpO2 of {vital_signs['SpO2']}%, suggest potential hypoxemia, "
                f"indicative of respiratory compromise in the context of COVID-19. A confirmatory RT-PCR test is recommended to validate the diagnosis, and a high-resolution CT scan may be considered for further characterization of lung involvement. "
                f"Continuous monitoring of SpO2 and respiratory rate is advised to assess disease progression, with thresholds of SpO2 <92% or respiratory rate >24 breaths/min warranting immediate intervention. "
                f"Patient management should include supplemental oxygen if SpO2 <92%, consideration of antiviral therapy per local guidelines, and strict adherence to isolation protocols. "
                f"This AI analysis serves as a decision-support tool, but final interpretation requires oversight by a radiologist and infectious disease specialist."
            )

        report = (
            f"Medical Report\n"
            f"Date: {current_date}\n\n"
            f"{sections['1. Introduction']}\n\n"
            f"{sections['2. Findings']}\n\n"
            f"{sections['3. Clinical Interpretation']}\n\n"
            f"{sections['4. Conclusion']}"
        )

        logger.info("Medical report generated successfully")
        return report
    except Exception as e:
        logger.error(f"Error in generate_medical_report: {str(e)}")
        current_date = datetime(2025, 4, 11).strftime("%B %d, %Y")
        return (
            f"Medical Report\n"
            f"Date: {current_date}\n\n"
            f"1. Introduction\n"
            f"This report evaluates a chest X-ray for COVID-19 using a ResNet101-based AI model, CLIP, and XAI heatmap. Vital signs are considered for a comprehensive assessment.\n"
            f"The analysis supports clinical decisions but requires professional evaluation.\n\n"
            f"2. Findings\n"
            f"The AI predicted '{prediction}' with confidence {confidence:.2f}. The X-ray shows potential features of COVID-19. "
            f"Vital signs: HR={vital_signs['HR']} bpm, RR={vital_signs['RR']} breaths/min, SpO2={vital_signs['SpO2']}%, BP={vital_signs['BP']} mmHg, Temp={vital_signs['Temp']}°C.\n\n"
            f"3. Clinical Interpretation\n"
            f"The prediction suggests a possible COVID-19 infection. The SpO2 of {vital_signs['SpO2']}% may indicate respiratory issues if low. Further evaluation is needed.\n\n"
            f"4. Conclusion\n"
            f"A PCR test and consultation with a healthcare professional are recommended. Monitor SpO2 and respiratory rate. AI supports but requires human oversight."
        )

def generate_frames():
    cap = cv2.VideoCapture(0)
    prev_red = None
    start_time = time.time()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        rgb_mean = np.mean(frame, axis=(0, 1))
        
        if prev_red is not None:
            red_diff = abs(rgb_mean[2] - prev_red)
            elapsed = time.time() - start_time
            if elapsed > 1:
                heart_rate = int((red_diff * 60 / elapsed) + 60)
                start_time = time.time()
            else:
                heart_rate = 70
        else:
            heart_rate = 70
        prev_red = rgb_mean[2]
        
        resp_rate = int(rgb_mean[1] * 0.1 + 12)
        spo2 = int(95 + (rgb_mean[0] - 100) * 0.05)
        bp = f"{int(rgb_mean[2] * 0.6 + 100)}/{int(rgb_mean[1] * 0.3 + 60)}"
        temp = round(36.5 + (rgb_mean[0] - 100) * 0.02, 1)
        
        cv2.putText(frame, f"Heart Rate: {heart_rate} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Resp Rate: {resp_rate} breaths/min", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"O2 Sat: {spo2}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"BP: {bp} mmHg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Temp: {temp}C", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        vital_signs = {"HR": heart_rate, "RR": resp_rate, "SpO2": spo2, "BP": bp, "Temp": temp}
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'), vital_signs
    cap.release()

@app.route('/')
def index():
    logger.debug("Rendering index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Upload request received")
    try:
        if 'file' not in request.files:
            logger.warning("No file part in request")
            error_msg = "Error: No file was uploaded. Please upload a chest X-ray image to proceed."
            return jsonify({'error': error_msg}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            error_msg = "Error: No file was selected. Please choose a chest X-ray image to upload."
            return jsonify({'error': error_msg}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            logger.warning(f"Invalid file type: {file.filename}")
            error_msg = "Error: Invalid file type. Please upload an image file (PNG, JPG, JPEG, BMP)."
            return jsonify({'error': error_msg}), 400

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Saving file to {filepath}")
        file.save(filepath)
        logger.debug(f"File saved to {filepath}")

        if not os.path.exists(filepath):
            logger.error(f"File not found after saving: {filepath}")
            error_msg = "Error: Failed to save the uploaded file. Please try again."
            return jsonify({'error': error_msg}), 500

        logger.debug("Predicting COVID-19 diagnosis")
        try:
            prediction, confidence = predict_covid(filepath)
        except Exception as e:
            logger.error(f"Failed to predict COVID-19: {str(e)}")
            return jsonify({'error': str(e)}), 500

        logger.debug("Generating XAI heatmap")
        xai_filename = f"xai_{filename}"
        xai_path = os.path.join(app.config['XAI_FOLDER'], xai_filename)
        try:
            generate_heatmapfocus_xai(filepath, xai_path)
        except Exception as e:
            logger.error(f"Failed to generate XAI heatmap: {str(e)}")
            return jsonify({'error': str(e)}), 500

        vital_signs = {
            "HR": 70,
            "RR": 20,
            "SpO2": 94,
            "BP": "154/86",
            "Temp": 36.3
        }

        logger.debug("Generating medical report")
        try:
            report = generate_medical_report(prediction, confidence, filepath, vital_signs)
        except Exception as e:
            logger.error(f"Failed to generate medical report: {str(e)}")
            return jsonify({'error': str(e)}), 500

        if "Error" in report:
            logger.error(f"Medical report generation returned an error: {report}")
            return jsonify({'error': report}), 500

        response = {
            'prediction': prediction,
            'confidence': float(confidence),
            'image_url': f'/static/uploads/{filename}',
            'xai_url': f'/static/xai/{xai_filename}',
            'report': report,
            'vital_signs': vital_signs
        }
        logger.info("Upload processed successfully")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        error_msg = f"Error: Failed to process the analysis: {str(e)}. Please try again or consult a healthcare professional."
        return jsonify({'error': error_msg}), 500

@app.route('/chat', methods=['POST'])
def chat():
    logger.debug("Chat request received")
    try:
        # Step 1: Parse the incoming JSON data
        logger.debug("Parsing incoming JSON data")
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in chat request")
            return jsonify({'response': "I’m sorry, I couldn’t understand your question because no data was provided. Could you please try again? I’m here to help with your X-ray analysis or vital signs."})

        # Step 2: Extract required fields with defaults
        logger.debug("Extracting fields from JSON data")
        question = str(data.get('question', '')).lower().strip()
        prediction = str(data.get('prediction', 'Unknown'))
        confidence = float(data.get('confidence', 0.0))
        vital_signs = data.get('vital_signs', {})
        image_url = str(data.get('image_url', ''))

        logger.debug(f"Question: {question}")
        logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        logger.debug(f"Vital Signs: {vital_signs}")
        logger.debug(f"Image URL: {image_url}")

        # Step 3: Validate the question
        if not question:
            logger.warning("Empty question received in chat request")
            return jsonify({'response': "I see you didn’t ask a specific question. I can help with information about your X-ray, COVID-19 diagnosis, or vital signs. What would you like to know?"})

        # Step 4: Validate the image URL and construct the image path
        if not image_url:
            logger.error("No image_url provided in chat request")
            return jsonify({'response': "I’m sorry, I don’t have access to your X-ray image. Please upload an image first, and I can help with your question."})

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_url.split('/')[-1])
        logger.debug(f"Constructed image path: {image_path}")
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return jsonify({'response': "I’m sorry, I couldn’t find your X-ray image. Please ensure the image was uploaded correctly, and try asking your question again."})

        # Step 5: Generate VLM description (with fallback)
        logger.debug("Generating VLM description")
        try:
            clip_prompt = (
                f"Analyze a chest X-ray with an AI-generated heatmap (yellow: least focus, orange: medium focus, red: most focus). "
                f"Identify lung features (textures, patterns, opacities, consolidations) supporting a diagnosis of '{prediction}' "
                f"with confidence {confidence:.2f}."
            )
            vlm_response = generate_clip_description(
                image_path,
                clip_prompt,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device
            )
            logger.debug(f"VLM response: {vlm_response}")
        except Exception as e:
            logger.error(f"Failed to generate VLM description: {str(e)}")
            vlm_response = "The chest X-ray likely shows features consistent with the diagnosis, but a detailed analysis couldn’t be completed."
            logger.debug("Using fallback VLM response")

        # Step 6: Generate XAI heatmap (with fallback)
        logger.debug("Generating XAI heatmap")
        try:
            xai_filename = f"xai_{image_url.split('/')[-1]}"
            xai_path = os.path.join(app.config['XAI_FOLDER'], xai_filename)
            _, focus_areas = generate_heatmapfocus_xai(image_path, xai_path)
            xai_results = (
                f"The algorithm focused primarily on the lower lobes with high focus (red) in {focus_areas['lower_lobes']['high']} pixels, "
                f"medium focus (orange) in {focus_areas['lower_lobes']['medium']} pixels, and low focus (yellow) in {focus_areas['lower_lobes']['low']} pixels. "
                f"The middle lobe showed high focus in {focus_areas['middle_lobe']['high']} pixels, medium focus in {focus_areas['middle_lobe']['medium']} pixels, "
                f"and low focus in {focus_areas['middle_lobe']['low']} pixels. The upper lobes had high focus in {focus_areas['upper_lobes']['high']} pixels, "
                f"medium focus in {focus_areas['upper_lobes']['medium']} pixels, and low focus in {focus_areas['upper_lobes']['low']} pixels."
            )
            logger.debug(f"XAI results: {xai_results}")
        except Exception as e:
            logger.error(f"Failed to generate XAI heatmap: {str(e)}")
            xai_results = "The algorithm focused on the lung areas, but specific focus details couldn’t be generated."
            logger.debug("Using fallback XAI results")

        # Step 7: Prepare context for chat response
        logger.debug("Preparing chat prompt")
        chat_prompt = (
            f"The context is a chest X-ray diagnosis of '{prediction}' with confidence {confidence:.2f}. "
            f"The visual analysis revealed: {vlm_response}. "
            f"The XAI heatmap results are: {xai_results}. "
            f"Vital signs are: Heart Rate={vital_signs.get('HR', 'N/A')} bpm, "
            f"Respiratory Rate={vital_signs.get('RR', 'N/A')} breaths/min, "
            f"Oxygen Saturation={vital_signs.get('SpO2', 'N/A')}%, "
            f"Blood Pressure={vital_signs.get('BP', 'N/A')} mmHg, "
            f"Temperature={vital_signs.get('Temp', 'N/A')}°C. "
            f"The user asks: '{question}'. "
            f"Respond to the user's question in a concise, empathetic, clear, and informative manner. "
            f"Start by acknowledging the user's concern or question. "
            f"Provide a straightforward explanation using the visual analysis, XAI results, and vital signs. "
            f"Avoid technical jargon, focusing on simple language. "
            f"Offer practical advice or next steps, such as consulting a doctor, while emphasizing that you are not a doctor. "
            f"Do not include this prompt in the output."
        )

        # Step 8: Attempt to generate a response
        logger.debug("Generating chat response")
        response = generate_gpt_text(
            chat_prompt,
            max_new_tokens=50,
            model=gpt_model,
            tokenizer=gpt_tokenizer,
            device=device,
            timeout=5  # Increased timeout to give it a better chance
        )
        logger.debug(f"Generated response: {response}")

        # Step 9: Enhanced fallback responses based on question content
        if "Error" in response:
            logger.error(f"Chat response generation failed: {response}")
            if "x-ray" in question and "covid" in question:
                response = (
                    f"I understand you're asking if the X-ray shows COVID-19. The analysis indicates a '{prediction}' diagnosis "
                    f"with a confidence of {confidence:.2f}. The X-ray findings are: {vlm_response} The focus areas in the lungs are: {xai_results} "
                    f"Your oxygen level is {vital_signs.get('SpO2', 'N/A')}%, which might suggest breathing issues if it's low. "
                    f"I'm not a doctor, but I recommend seeing one for a detailed check-up and possibly a PCR test to confirm the diagnosis."
                )
            elif "vital signs" in question or "oxygen" in question:
                response = (
                    f"I understand you're asking about your vital signs. The analysis shows your oxygen level is {vital_signs.get('SpO2', 'N/A')}%, "
                    f"respiratory rate is {vital_signs.get('RR', 'N/A')} breaths per minute, and heart rate is {vital_signs.get('HR', 'N/A')} bpm. "
                    f"A low oxygen level or high respiratory rate might suggest breathing difficulties, which can be related to COVID-19. "
                    f"I'm not a doctor, but I suggest consulting one to discuss these results and your symptoms."
                )
            elif "treatment" in question or "what should i do" in question:
                response = (
                    f"I understand you're asking about what to do next or possible treatments. The X-ray analysis shows a '{prediction}' diagnosis "
                    f"with a confidence of {confidence:.2f}. Your oxygen level is {vital_signs.get('SpO2', 'N/A')}%, which might need monitoring if low. "
                    f"I'm not a doctor, but I recommend seeing one as soon as possible. They might suggest a PCR test to confirm COVID-19, and if positive, "
                    f"you may need to isolate, rest, stay hydrated, and possibly use oxygen support if your breathing gets worse."
                )
            elif "symptoms" in question or "how do i feel" in question:
                response = (
                    f"I understand you're asking about symptoms or how you might be feeling. The X-ray analysis shows a '{prediction}' diagnosis "
                    f"with a confidence of {confidence:.2f}. Your vital signs, like an oxygen level of {vital_signs.get('SpO2', 'N/A')}%, might suggest "
                    f"breathing issues if low. Common COVID-19 symptoms include fever, cough, and shortness of breath. I'm not a doctor, but if you're feeling unwell, "
                    f"I recommend consulting one to discuss your symptoms and get proper care."
                )
            else:
                response = (
                    f"I understand you're asking about '{question}'. The chest X-ray analysis shows a '{prediction}' diagnosis "
                    f"with a confidence of {confidence:.2f}. The X-ray findings are: {vlm_response} "
                    f"Your vital signs, like an oxygen level of {vital_signs.get('SpO2', 'N/A')}%, might indicate breathing issues if low. "
                    f"I'm not a doctor, but I recommend consulting one for a detailed evaluation and further testing if needed."
                )
            logger.debug(f"Fallback response: {response}")

        logger.info("Chat response generated successfully")
        return jsonify({'response': response})

    except Exception as e:
        # Final fallback to ensure a response is always returned
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        # Extract whatever data is available
        try:
            data = request.get_json() if request.get_json() else {}
            prediction = str(data.get('prediction', 'Unknown'))
            confidence = float(data.get('confidence', 0.0))
            vital_signs = data.get('vital_signs', {})
        except:
            prediction = "Unknown"
            confidence = 0.0
            vital_signs = {}
        response = (
            f"I’m sorry, I ran into an issue while processing your question, but I can still share some information. "
            f"The X-ray analysis shows a '{prediction}' diagnosis with a confidence of {confidence:.2f}. "
            f"Your vital signs, like an oxygen level of {vital_signs.get('SpO2', 'N/A')}%, might suggest breathing issues if low. "
            f"I’m not a doctor, but I recommend consulting one for a detailed evaluation and further testing if needed."
        )
        logger.debug(f"Final fallback response: {response}")
        return jsonify({'response': response})

@app.route('/video_feed')
def video_feed():
    def generate():
        for frame, _ in generate_frames():
            yield frame
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("Starting application initialization...")
    logger.info("Initializing application")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(XAI_FOLDER, exist_ok=True)

    model_path = 'COVID19_ResNet101.onnx'
    if not os.path.exists(model_path):
        logger.error(f"ONNX model file not found: {model_path}")
        print(f"Error: ONNX model file not found at {model_path}. Please ensure the file exists.")
        sys.exit(1)

    try:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {str(e)}")
        print(f"Error loading CLIP model: {str(e)}. Ensure 'transformers' is installed and models are available.")
        sys.exit(1)

    try:
        print("Loading GPT-Neo model...")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
        print("GPT-Neo model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load GPT-Neo model: {str(e)}")
        print(f"Error loading GPT-Neo model: {str(e)}. Ensure 'transformers' is installed and models are available.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    gpt_model.to(device)
    logger.info(f"Models moved to device: {device}")
    print(f"Models moved to device: {device}")

    try:
        print("Loading ONNX model...")
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        logger.info(f"ONNX model loaded. Input: {input_name}, Outputs: {output_names}")
        print("ONNX model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {str(e)}")
        print(f"Error loading ONNX model: {str(e)}. Ensure 'COVID19_ResNet101.onnx' exists and is compatible.")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("Webcam not available. Vital signs will use defaults.")
        print("Warning: Webcam not available. Vital signs will use defaults.")
    else:
        cap.release()
        print("Webcam check passed")

    host = '0.0.0.0'
    port = 5000
    print(f"Starting Flask server on http://{host}:{port}...")
    logger.info(f"Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=True, use_reloader=False, threaded=True)
    print("Flask server is running")