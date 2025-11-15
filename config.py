"""
Configuration file for gender bias detection project
"""

# Models to compare
# Models to compare
MODELS = [          # 124M parameters (already done)
    "gpt2-medium",    # 355M parameters
]

# Occupation terms (30 words as per MVP)
OCCUPATION_TERMS = [
    # High-prestige occupations
    "doctor", "engineer", "scientist", "professor", "lawyer",
    "CEO", "manager", "director", "architect", "surgeon",
    
    # Mid-prestige occupations
    "teacher", "accountant", "programmer", "analyst", "designer",
    "journalist", "consultant", "pharmacist", "pilot", "dentist",
    
    # Lower-prestige occupations (to avoid bias in selection)
    "nurse", "secretary", "assistant", "receptionist", "cashier",
    "cleaner", "cook", "waiter", "hairdresser", "babysitter"
]

# Gender anchor terms
MALE_TERMS = ["man", "he"]
FEMALE_TERMS = ["woman", "she"]
NEUTRAL_TERMS = ["they"]  # For future analysis

# All gender terms combined
GENDER_TERMS = MALE_TERMS + FEMALE_TERMS + NEUTRAL_TERMS

# Known validation examples (from literature)
VALIDATION_EXAMPLES = {
    "doctor": "male",  # Expected to lean male
    "nurse": "female",  # Expected to lean female
    "engineer": "male",
    "secretary": "female",
    "CEO": "male"
}

# Analysis parameters
COSINE_THRESHOLD = 0.1  # Minimum difference to consider meaningful
WEAT_EFFECT_SIZE_THRESHOLD = 0.1

# File paths
DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"