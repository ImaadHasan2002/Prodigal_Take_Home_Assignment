import re
import requests
import os
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json

# Using  language model for toxicity detection
model_name = "meta-llama/Llama-2-7b-chat-hf"
llm_model = AutoModelForCausalLM.from_pretrained(model_name)
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Fallback profanity list (kept for baseline filtering)
PROFANE_WORDS = [
    'ass', 'damn', 'hell', 'shit', 'fuck', 'bitch', 'crap', 'bastard', 'asshole',
    'cock', 'dick', 'pussy', 'whore', 'slut', 'piss', 'bloody', 'cunt', 'bollocks',
    'fag', 'faggot', 'nigger', 'nigga', 'retard', 'motherfucker'
]

def regex_detection(text):
    """Fallback regex-based profanity detection"""
    text = text.lower()
    pattern = r'\b(' + '|'.join(re.escape(word) for word in PROFANE_WORDS) + r')\b'
    matches = re.findall(pattern, text)
    return bool(matches), matches

def llm_toxicity_detection(text):
    """
    Detect toxicity using a large language model with multiple prompting strategies
    
    Args:
        text (str): Text to analyze for toxicity
    
    Returns:
        tuple: (is_toxic, details)
    """
    # Multiple prompting strategies for robust detection
    toxicity_prompts = [
        f"Is the following text highly offensive or toxic? Respond with a JSON object containing 'is_toxic' (boolean) and 'reason' (string): {text}",
        f"Carefully analyze the text for potentially harmful language. Provide a JSON response with toxicity assessment: {text}",
        f"Evaluate the toxicity level of this text. Respond strictly with a JSON object with 'is_toxic' and explain why: {text}"
    ]
    
    for prompt in toxicity_prompts:
        try:
            # Tokenize and generate response
            inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            outputs = llm_model.generate(
                **inputs, 
                max_new_tokens=100, 
                num_return_sequences=1,
                temperature=0.2
            )
            
            # Decode response
            response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    if isinstance(result.get('is_toxic'), bool):
                        return result.get('is_toxic'), result
                except json.JSONDecodeError:
                    continue
        
        except Exception as e:
            print(f"LLM detection error: {str(e)}")
    
    # Fallback to regex if LLM fails
    regex_result, matches = regex_detection(text)
    return regex_result, {"is_toxic": regex_result, "method": "regex", "matches": matches}

def check_profanity_in_text(text, approach="llm"):
    """
    Check profanity in text with flexible detection approach
    
    Args:
        text (str): Text to check
        approach (str): Detection method ('llm', 'regex')
    
    Returns:
        bool: Whether text is considered toxic
    """
    approach = approach.lower()
    if approach == "llm":
        return llm_toxicity_detection(text)[0]
    elif approach == "regex":
        return regex_detection(text)[0]
    else:
        print(f"Invalid approach '{approach}'. Supported approaches are 'llm' and 'regex'.")
        return False

def analyze(call_data, approach="llm"):
    """
    Analyze call data for profanity with enhanced detection
    
    Args:
        call_data (list): List of call utterances
        approach (str): Detection method
    
    Returns:
        dict: Profanity status for agent and borrower
    """
    agent_profanity = False
    borrower_profanity = False
    
    for utterance in call_data:
        speaker = utterance.get("speaker", "").strip().lower()
        text = utterance.get("text", "")
        
        if speaker == "agent":
            if check_profanity_in_text(text, approach):
                agent_profanity = True
        elif speaker in {"borrower", "customer"}:
            if check_profanity_in_text(text, approach):
                borrower_profanity = True
    
    return {
        "agent_profanity": agent_profanity,
        "borrower_profanity": borrower_profanity
    }