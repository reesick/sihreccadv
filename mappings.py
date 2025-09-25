# ====================================================================
# PERSONALITY TO CAREER KEYWORDS MAPPING
# Maps Big 5 and RIASEC scores to relevant search keywords
# ====================================================================

# Big 5 Personality Traits → Career Keywords
BIG5_KEYWORDS = {
    "extraversion": [
        "collaborative", "team", "social", "communication", "leadership",
        "networking", "public speaking", "client interaction", "group work",
        "interpersonal", "customer service", "sales", "management"
    ],
    
    "agreeableness": [
        "helping", "supportive", "cooperative", "counseling", "healthcare",
        "social work", "teaching", "mentoring", "caregiving", "humanitarian",
        "community", "non-profit", "volunteer", "empathy"
    ],
    
    "conscientiousness": [
        "organized", "detailed", "systematic", "planning", "methodical",
        "precise", "quality control", "project management", "structured",
        "reliable", "thorough", "documentation", "process improvement"
    ],
    
    "neuroticism": [
        # Note: High neuroticism suggests avoiding high-stress roles
        # Low neuroticism (emotional stability) keywords:
        "stable", "calm", "stress management", "crisis handling", "resilient",
        "emergency", "high pressure", "decision making", "leadership"
    ],
    
    "openness": [
        "creative", "innovative", "artistic", "design", "research",
        "experimental", "flexible", "adaptable", "learning", "exploration",
        "technology", "new ideas", "problem solving", "intellectual"
    ]
}

# RIASEC Interest Types → Career Keywords
RIASEC_KEYWORDS = {
    "realistic": [
        "hands-on", "technical", "mechanical", "building", "construction",
        "engineering", "manufacturing", "tools", "equipment", "repair",
        "outdoors", "physical", "practical", "tangible", "operations",
        "maintenance", "installation", "production", "machinery"
    ],
    
    "investigative": [
        "research", "analysis", "data", "science", "laboratory", "testing",
        "investigation", "problem solving", "analytical", "mathematics",
        "statistics", "experiments", "theories", "discovery", "knowledge",
        "scientific", "systematic study", "hypothesis", "evidence"
    ],
    
    "artistic": [
        "creative", "design", "art", "music", "writing", "performance",
        "visual", "aesthetic", "imagination", "expression", "media",
        "entertainment", "culture", "literature", "graphics", "photography",
        "theater", "film", "advertising", "marketing creative"
    ],
    
    "social": [
        "helping", "teaching", "counseling", "healthcare", "education",
        "social work", "therapy", "training", "mentoring", "community",
        "human services", "psychology", "development", "support",
        "guidance", "rehabilitation", "social services", "care"
    ],
    
    "enterprising": [
        "business", "management", "leadership", "entrepreneurship", "sales",
        "marketing", "finance", "negotiation", "strategy", "operations",
        "administration", "consulting", "business development", "executive",
        "commerce", "trade", "economics", "investment", "profit"
    ],
    
    "conventional": [
        "organized", "systematic", "data entry", "administration", "clerical",
        "accounting", "bookkeeping", "records", "filing", "procedures",
        "office", "structured", "routine", "detailed", "compliance",
        "documentation", "process", "standards", "regulations"
    ]
}

# Score interpretation thresholds
SCORE_THRESHOLDS = {
    "high": 4.0,      # Strong trait/interest - include in search
    "medium": 3.0,    # Moderate trait/interest - consider including
    "low": 2.0        # Weak trait/interest - exclude from search
}

# Maximum keywords to extract per trait/interest
MAX_KEYWORDS_PER_TRAIT = 5

# Weight for combining multiple traits
TRAIT_WEIGHTS = {
    "primary": 1.0,     # Highest scoring trait gets full weight
    "secondary": 0.7,   # Second highest gets 70% weight
    "tertiary": 0.5     # Third highest gets 50% weight
}

def get_trait_level(score: float) -> str:
    """
    Determine trait level based on score
    
    Args:
        score: Personality trait score (1.0-5.0)
        
    Returns:
        str: 'high', 'medium', or 'low'
    """
    if score >= SCORE_THRESHOLDS["high"]:
        return "high"
    elif score >= SCORE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"