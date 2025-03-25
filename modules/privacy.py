import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Patterns for sensitive information
BALANCE_PATTERN = r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s+dollars'
ACCOUNT_PATTERN = r'account\s+(?:number|#)?\s*\d+|account\s+(?:number|#)?\s*[A-Z0-9]{4,}'
SSN_PATTERN = r'\d{3}[-\s]?\d{2}[-\s]?\d{4}'
DOB_PATTERN = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
ADDRESS_PATTERN = r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|circle|cir|court|ct|place|pl|way)'

# Keywords/phrases for verification
VERIFICATION_KEYWORDS = [
    'verify', 'verification', 'confirm', 'identity', 'security', 
    'validate', 'authentication', 'date of birth', 'social security', 
    'address', 'zip code', 'last four'
]

def regex_check_verification(conversation):
    """Check if agent properly verified identity before sharing sensitive info using regex"""
    verification_done = False
    sensitive_info_shared = False
    verification_utterances = []
    sensitive_utterances = []
    problematic_utterances = []
    
    sensitive_info_types = {
        "Balance information": False,
        "Account details": False
    }
    
    for i, utterance in enumerate(conversation):
        text = utterance["text"].lower()
        speaker = utterance["speaker"].lower()
        
        # Check for verification attempts
        if speaker == "agent" and any(keyword in text for keyword in VERIFICATION_KEYWORDS):
            verification_utterances.append(utterance)
            
            # Check if the verification was successful based on customer responses
            if i < len(conversation) - 1 and conversation[i+1]["speaker"].lower() in ["borrower", "customer"]:
                next_text = conversation[i+1]["text"].lower()
                if (re.search(DOB_PATTERN, next_text) or 
                    re.search(SSN_PATTERN, next_text) or 
                    re.search(ADDRESS_PATTERN, next_text)):
                    verification_done = True
        
        # Check for direct verification (SSN, DOB, address)
        if speaker == "borrower" or speaker == "customer":
            if (re.search(DOB_PATTERN, text) or 
                re.search(SSN_PATTERN, text) or 
                re.search(ADDRESS_PATTERN, text)):
                verification_done = True
                verification_utterances.append(utterance)
                
        # Check for sensitive info sharing by agent
        if speaker == "agent":
            if re.search(BALANCE_PATTERN, text):
                sensitive_info_shared = True
                sensitive_info_types["Balance information"] = True
                sensitive_utterances.append(utterance)
                
                if not verification_done:
                    problematic_utterances.append(utterance)
                    
            if re.search(ACCOUNT_PATTERN, text):
                sensitive_info_shared = True
                sensitive_info_types["Account details"] = True
                sensitive_utterances.append(utterance)
                
                if not verification_done:
                    problematic_utterances.append(utterance)
    
    return {
        "verified": verification_done,
        "sensitive_info_shared": sensitive_info_types,
        "verification_utterances": verification_utterances,
        "sensitive_utterances": sensitive_utterances,
        "problematic_utterances": problematic_utterances,
        "privacy_violation": sensitive_info_shared and not verification_done
    }

def ai_check_verification(conversation):
    """
    Check for identity verification using a more sophisticated approach
    This implementation is a more advanced version of the regex approach
    In a real-world scenario, this would use an actual ML/LLM model
    """
    # Begin by extracting all text and organizing by speaker
    agent_texts = []
    customer_texts = []
    
    for utterance in conversation:
        if utterance["speaker"].lower() == "agent":
            agent_texts.append(utterance["text"])
        else:
            customer_texts.append(utterance["text"])
    
    # Check for verification requests
    verification_attempts = []
    for i, utterance in enumerate(conversation):
        if utterance["speaker"].lower() == "agent":
            text = utterance["text"].lower()
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # Look for verification questions or requests
                if any(keyword in sentence for keyword in VERIFICATION_KEYWORDS):
                    if "?" in sentence or any(verb in sentence for verb in ["confirm", "verify", "tell", "provide"]):
                        verification_attempts.append((i, utterance))
    
    # Check for customer responses to verification requests
    verification_responses = []
    for attempt_idx, attempt_utterance in verification_attempts:
        if attempt_idx + 1 < len(conversation) and conversation[attempt_idx + 1]["speaker"].lower() != "agent":
            response = conversation[attempt_idx + 1]
            verification_responses.append(response)
    
    # Determine if verification was successful
    verification_done = False
    for response in verification_responses:
        text = response["text"].lower()
        # Check for SSN, DOB, or address in response
        if (re.search(SSN_PATTERN, text) or 
            re.search(DOB_PATTERN, text) or 
            re.search(ADDRESS_PATTERN, text)):
            verification_done = True
            break
    
    # Look for sensitive information shared by agent
    sensitive_info_types = {
        "Balance information": False,
        "Account details": False
    }
    
    sensitive_utterances = []
    problematic_utterances = []
    
    # Initialize verification status tracking
    verification_utterances = [attempt[1] for attempt in verification_attempts]
    verification_utterances.extend(verification_responses)
    
    # Check for sharing of sensitive information
    for utterance in conversation:
        if utterance["speaker"].lower() == "agent":
            text = utterance["text"]
            
            # Check for balance information
            if re.search(BALANCE_PATTERN, text):
                sensitive_info_types["Balance information"] = True
                sensitive_utterances.append(utterance)
                
                if not verification_done:
                    problematic_utterances.append(utterance)
            
            # Check for account details
            if re.search(ACCOUNT_PATTERN, text):
                sensitive_info_types["Account details"] = True
                sensitive_utterances.append(utterance)
                
                if not verification_done:
                    problematic_utterances.append(utterance)
    
    privacy_violation = (sensitive_info_types["Balance information"] or 
                         sensitive_info_types["Account details"]) and not verification_done
    
    return {
        "verified": verification_done,
        "sensitive_info_shared": sensitive_info_types,
        "verification_utterances": verification_utterances,
        "sensitive_utterances": sensitive_utterances,
        "problematic_utterances": problematic_utterances,
        "privacy_violation": privacy_violation
    }

def get_verification_status(call_data):
    """Get detailed verification status for display purposes"""
    return regex_check_verification(call_data)

def analyze(call_data, approach="Regex"):
    """
    Analyze call for privacy/compliance violations
    Returns: {
        "privacy_violation": bool
    }
    """
    if approach == "Regex":
        results = regex_check_verification(call_data)
    else:
        results = ai_check_verification(call_data)
    
    return {
        "privacy_violation": results["privacy_violation"]
    }