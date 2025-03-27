import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# Load sentiment analysis model for context-aware detection
try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Existing patterns for sensitive information
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

# Enhanced sensitive information detection using NER
def detect_sensitive_info_ner(text):
    """Detect sensitive information using Named Entity Recognition"""
    doc = nlp(text)
    sensitive_entities = {
        "PERSON": False,
        "DATE": False,
        "MONEY": False,
        "GPE": False,  # Geopolitical Entity (addresses)
        "CARDINAL": False  # Numbers that might be account numbers or SSN
    }
    
    for ent in doc.ents:
        if ent.label_ in sensitive_entities:
            sensitive_entities[ent.label_] = True
    
    return sensitive_entities

def analyze_sentiment(text):
    """Analyze the sentiment of text to detect aggressive or negative tone"""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    scores = scores.numpy()
    
    # Returns negative sentiment score (higher means more negative)
    return float(scores[0][0])

def hybrid_check_verification(conversation):
    """
    Enhanced verification check using both regex and NER approaches
    """
    verification_done = False
    sensitive_info_shared = False
    verification_utterances = []
    sensitive_utterances = []
    problematic_utterances = []
    
    sensitive_info_types = {
        "Balance information": False,
        "Account details": False,
        "Personal information": False,
        "Address information": False
    }
    
    # Track conversation sentiment and context
    conversation_sentiment = []
    
    for i, utterance in enumerate(conversation):
        text = utterance["text"]
        text_lower = text.lower()
        speaker = utterance["speaker"].lower()
        
        # Analyze sentiment for context awareness
        sentiment_score = analyze_sentiment(text)
        conversation_sentiment.append(sentiment_score)
        
        # Check for verification attempts
        if speaker == "agent" and any(keyword in text_lower for keyword in VERIFICATION_KEYWORDS):
            verification_utterances.append(utterance)
            
            # Check if the verification was successful based on customer responses
            if i < len(conversation) - 1 and conversation[i+1]["speaker"].lower() in ["borrower", "customer"]:
                next_text = conversation[i+1]["text"].lower()
                
                # Check with regex patterns
                if (re.search(DOB_PATTERN, next_text) or 
                    re.search(SSN_PATTERN, next_text) or 
                    re.search(ADDRESS_PATTERN, next_text)):
                    verification_done = True
                
                # Check with NER
                ner_entities = detect_sensitive_info_ner(next_text)
                if ner_entities["PERSON"] or ner_entities["DATE"] or ner_entities["GPE"]:
                    verification_done = True
        
        # Check for direct verification (SSN, DOB, address)
        if speaker in ["borrower", "customer"]:
            # Check with regex
            if (re.search(DOB_PATTERN, text_lower) or 
                re.search(SSN_PATTERN, text_lower) or 
                re.search(ADDRESS_PATTERN, text_lower)):
                verification_done = True
                verification_utterances.append(utterance)
            
            # Check with NER
            ner_entities = detect_sensitive_info_ner(text)
            if ner_entities["PERSON"] or ner_entities["DATE"] or ner_entities["GPE"]:
                verification_done = True
                verification_utterances.append(utterance)
                
        # Check for sensitive info sharing by agent
        if speaker == "agent":
            # Check with regex
            if re.search(BALANCE_PATTERN, text_lower):
                sensitive_info_shared = True
                sensitive_info_types["Balance information"] = True
                sensitive_utterances.append(utterance)
                
                if not verification_done:
                    problematic_utterances.append(utterance)
                    
            if re.search(ACCOUNT_PATTERN, text_lower):
                sensitive_info_shared = True
                sensitive_info_types["Account details"] = True
                sensitive_utterances.append(utterance)
                
                if not verification_done:
                    problematic_utterances.append(utterance)
            
            # Check with NER
            ner_entities = detect_sensitive_info_ner(text)
            
            if ner_entities["MONEY"]:
                sensitive_info_shared = True
                sensitive_info_types["Balance information"] = True
                if utterance not in sensitive_utterances:
                    sensitive_utterances.append(utterance)
                
                if not verification_done and utterance not in problematic_utterances:
                    problematic_utterances.append(utterance)
            
            if ner_entities["CARDINAL"]:
                sensitive_info_shared = True
                sensitive_info_types["Account details"] = True
                if utterance not in sensitive_utterances:
                    sensitive_utterances.append(utterance)
                
                if not verification_done and utterance not in problematic_utterances:
                    problematic_utterances.append(utterance)
            
            if ner_entities["PERSON"]:
                sensitive_info_shared = True
                sensitive_info_types["Personal information"] = True
                if utterance not in sensitive_utterances:
                    sensitive_utterances.append(utterance)
                
                if not verification_done and utterance not in problematic_utterances:
                    problematic_utterances.append(utterance)
            
            if ner_entities["GPE"]:
                sensitive_info_shared = True
                sensitive_info_types["Address information"] = True
                if utterance not in sensitive_utterances:
                    sensitive_utterances.append(utterance)
                
                if not verification_done and utterance not in problematic_utterances:
                    problematic_utterances.append(utterance)
    
    # Analyze conversation sentiment context
    avg_sentiment = sum(conversation_sentiment) / len(conversation_sentiment) if conversation_sentiment else 0
    high_negative_sentiment = any(score > 0.8 for score in conversation_sentiment)
    
    return {
        "verified": verification_done,
        "sensitive_info_shared": sensitive_info_types,
        "verification_utterances": verification_utterances,
        "sensitive_utterances": sensitive_utterances,
        "problematic_utterances": problematic_utterances,
        "privacy_violation": sensitive_info_shared and not verification_done,
        "sentiment_analysis": {
            "average_sentiment": avg_sentiment,
            "high_negative_detected": high_negative_sentiment,
            "sentiment_scores": conversation_sentiment
        }
    }

# Keep original functions for backward compatibility
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
    # Now this just calls our new hybrid approach
    return hybrid_check_verification(conversation)

def get_verification_status(call_data):
    """Get detailed verification status for display purposes"""
    return hybrid_check_verification(call_data)

def analyze(call_data, approach="Hybrid"):
    """
    Analyze call for privacy/compliance violations
    Returns: {
        "privacy_violation": bool,
        "sentiment_analysis": dict (if approach is Hybrid)
    }
    """
    if approach == "Regex":
        results = regex_check_verification(call_data)
        return {
            "privacy_violation": results["privacy_violation"]
        }
    elif approach == "AI":
        results = ai_check_verification(call_data)
        return {
            "privacy_violation": results["privacy_violation"]
        }
    else:  # Hybrid approach (default)
        results = hybrid_check_verification(call_data)
        return {
            "privacy_violation": results["privacy_violation"],
            "sentiment_analysis": results["sentiment_analysis"]
        }