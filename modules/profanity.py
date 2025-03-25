import re
import requests
import os
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Common profanity words list (simplified for demonstration)
PROFANE_WORDS = [
    'ass', 'damn', 'hell', 'shit', 'fuck', 'bitch', 'crap', 'bastard', 'asshole',
    'cock', 'dick', 'pussy', 'whore', 'slut', 'piss', 'bloody', 'cunt', 'bollocks',
    'fag', 'faggot', 'nigger', 'nigga', 'retard', 'motherfucker'
]

def regex_detection(text):
    """Detect profanity using regex patterns"""
    text = text.lower()
    # Create a pattern to match whole words only
    pattern = r'\b(' + '|'.join(PROFANE_WORDS) + r')\b'
    matches = re.findall(pattern, text)
    return len(matches) > 0, matches

def ai_detection(text):
    """
    Detect profanity using AI/LLM approach
    This is a simplified implementation - in a real scenario, you might use:
    1. A pre-trained sentiment/toxicity model 
    2. API call to a service like OpenAI, Content Moderation API, etc.
    """
    try:
        # Option 1: Use a local approach (similar to regex but more sophisticated)
        text_tokens = word_tokenize(text.lower())
        # Look for profane words and nearby context
        profanity_found = False
        matches = []
        
        for i, token in enumerate(text_tokens):
            if token in PROFANE_WORDS:
                profanity_found = True
                # Get context (up to 3 words before and after)
                start = max(0, i-3)
                end = min(len(text_tokens), i+4)
                context = ' '.join(text_tokens[start:end])
                matches.append(context)
        
        # Option 2: If we have an API key for OpenAI or similar service
        # Uncomment the section below and add your API key to use it
        """
        API_KEY = os.getenv("OPENAI_API_KEY")
        if API_KEY:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "text-moderation-latest",
                "input": text
            }
            
            response = requests.post(
                "https://api.openai.com/v1/moderations",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["results"][0]["flagged"], []
        """
        
        return profanity_found, matches
    
    except Exception as e:
        print(f"Error in AI detection: {str(e)}")
        # Fall back to regex if AI approach fails
        return regex_detection(text)

def check_profanity_in_text(text, approach="Regex"):
    """Check if text contains profanity using the specified approach"""
    if approach == "Regex":
        has_profanity, _ = regex_detection(text)
    else:
        has_profanity, _ = ai_detection(text)
    
    return has_profanity

def analyze(call_data, approach="Regex"):
    """
    Analyze call data for profanity
    Returns: {
        "agent_profanity": bool,
        "borrower_profanity": bool
    }
    """
    agent_profanity = False
    borrower_profanity = False
    
    for utterance in call_data:
        speaker = utterance["speaker"].lower()
        text = utterance["text"]
        
        if speaker == "agent":
            if check_profanity_in_text(text, approach):
                agent_profanity = True
        elif speaker == "borrower" or speaker == "customer":
            if check_profanity_in_text(text, approach):
                borrower_profanity = True
    
    return {
        "agent_profanity": agent_profanity,
        "borrower_profanity": borrower_profanity
    }