# ====================================================================
# PERSONALITY-BASED CAREER RECOMMENDER SYSTEM
# Complete FastAPI Application with Supabase + FAISS Integration
# ====================================================================

import pandas as pd
import numpy as np
import faiss
import os
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import uvicorn
from mappings import BIG5_KEYWORDS, RIASEC_KEYWORDS, SCORE_THRESHOLDS, get_trait_level

# ====================================================================
# PYDANTIC MODELS
# ====================================================================

class Big5Scores(BaseModel):
    extraversion: float = Field(..., ge=1.0, le=5.0)
    agreeableness: float = Field(..., ge=1.0, le=5.0)
    conscientiousness: float = Field(..., ge=1.0, le=5.0)
    neuroticism: float = Field(..., ge=1.0, le=5.0)
    openness: float = Field(..., ge=1.0, le=5.0)

class RiasecScores(BaseModel):
    realistic: float = Field(..., ge=1.0, le=5.0)
    investigative: float = Field(..., ge=1.0, le=5.0)
    artistic: float = Field(..., ge=1.0, le=5.0)
    social: float = Field(..., ge=1.0, le=5.0)
    enterprising: float = Field(..., ge=1.0, le=5.0)
    conventional: float = Field(..., ge=1.0, le=5.0)

class PersonalityRequest(BaseModel):
    big5_scores: Big5Scores
    riasec_scores: RiasecScores

class CareerResult(BaseModel):
    id: str
    title: str
    summary: str
    match_score: int
    salary_range: str
    education: str
    industry: str
    work_environment: str
    key_skills: List[str]
    personality_fit: Dict[str, Any]
    riasec_match: str
    location_options: str

class PersonalityResponse(BaseModel):
    status: str
    query_processed: str
    total_results: int
    processing_time_ms: int
    personality_profile: Dict[str, List[str]]
    results: List[CareerResult]
    filters_applied: Dict[str, Any]

# ====================================================================
# FASTAPI APP SETUP
# ====================================================================

app = FastAPI(
    title="Personality Career Recommender API",
    description="AI-powered career recommendations based on Big 5 and RIASEC personality scores",
    version="1.0.0"
)

# Global variables for caching
supabase_client: Optional[Client] = None
careers_df: Optional[pd.DataFrame] = None
embedding_model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.IndexFlatL2] = None
career_embeddings: Optional[np.ndarray] = None

# ====================================================================
# SUPABASE CONNECTION
# ====================================================================

def setup_supabase_connection() -> Client:
    """Setup Supabase connection using environment variables"""
    try:
        SUPABASE_URL = "https://scleqgoyuqpmjrkldtdu.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNjbGVxZ295dXFwbWpya2xkdGR1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjIwNzQ3OSwiZXhwIjoyMDcxNzgzNDc5fQ.Nejd9jE2NXd11267QgEyHajWSuZWZiKEMlvIkT_Ixdw"
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
        
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connection established")
        return client
        
    except Exception as e:
        print(f"❌ Error setting up Supabase: {str(e)}")
        raise

def load_careers_data(client: Client) -> pd.DataFrame:
    """Load careers data from Supabase"""
    try:
        print("📊 Loading careers data from Supabase...")
        response = client.table('carrers_db').select('*').execute()
        
        if not response.data:
            raise ValueError("No data found in carrers_db table")
        
        df = pd.DataFrame(response.data)
        print(f"✅ Loaded {len(df)} career records")
        return df
        
    except Exception as e:
        print(f"❌ Error loading careers data: {str(e)}")
        raise

# ====================================================================
# EMBEDDING & FAISS SETUP
# ====================================================================

def initialize_embedding_model() -> SentenceTransformer:
    """Initialize sentence transformer model"""
    try:
        print("🤖 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {str(e)}")
        raise

def generate_career_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for career descriptions"""
    try:
        print("⚡ Generating career embeddings...")
        descriptions = df['description'].fillna('No description available').tolist()
        embeddings = model.encode(descriptions, show_progress_bar=True)
        print(f"✅ Generated {embeddings.shape[0]} embeddings")
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        raise

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build FAISS index for similarity search"""
    try:
        print("🔍 Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        print(f"✅ FAISS index built with {index.ntotal} vectors")
        return index
    except Exception as e:
        print(f"❌ Error building FAISS index: {str(e)}")
        raise

# ====================================================================
# PERSONALITY SCORE PROCESSING
# ====================================================================

def convert_scores_to_query(big5_scores: Dict[str, float], riasec_scores: Dict[str, float]) -> str:
    """Convert personality scores to search query string"""
    try:
        keywords = []
        
        # Process RIASEC scores (interests - primary driver)
        riasec_sorted = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 3 RIASEC interests
        for i, (interest, score) in enumerate(riasec_sorted[:3]):
            if score >= SCORE_THRESHOLDS["medium"]:
                interest_keywords = RIASEC_KEYWORDS.get(interest, [])
                # Weight keywords by rank (top interest gets more keywords)
                num_keywords = max(1, 6 - i * 2)  # 6, 4, 2 keywords
                keywords.extend(interest_keywords[:num_keywords])
        
        # Process Big 5 scores (work style modifiers)
        for trait, score in big5_scores.items():
            trait_level = get_trait_level(score)
            
            if trait_level == "high":
                trait_keywords = BIG5_KEYWORDS.get(trait, [])
                # Add fewer keywords from Big 5 (they're modifiers)
                keywords.extend(trait_keywords[:3])
        
        # Create search query
        query = " ".join(keywords[:15])  # Limit to 15 keywords max
        return query
        
    except Exception as e:
        print(f"❌ Error converting scores to query: {str(e)}")
        return "general career opportunities"

def calculate_personality_match(career_row: pd.Series, big5_scores: Dict[str, float], riasec_scores: Dict[str, float]) -> Dict[str, Any]:
    """Calculate how well a career matches the user's personality"""
    try:
        # Get career's RIASEC codes
        career_riasec = str(career_row.get('riasec', '')).lower()
        career_personality = str(career_row.get('personality', '')).lower()
        
        # Calculate RIASEC match
        riasec_match_score = 0
        primary_match = None
        secondary_match = None
        
        # Find user's top 2 RIASEC interests
        user_riasec_sorted = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)
        top_interests = [item[0] for item in user_riasec_sorted[:2]]
        
        # Check if career matches user's interests
        for i, interest in enumerate(top_interests):
            if interest[:4].lower() in career_riasec:  # Match first 4 chars
                match_weight = 1.0 if i == 0 else 0.6  # Primary vs secondary match
                riasec_match_score += user_riasec_sorted[i][1] * match_weight
                
                if i == 0:
                    primary_match = interest.title()
                else:
                    secondary_match = interest.title()
        
        # Calculate personality trait match
        personality_match_score = 0
        for trait, score in big5_scores.items():
            if trait.lower() in career_personality and score >= SCORE_THRESHOLDS["high"]:
                personality_match_score += score * 0.5
        
        # Combined compatibility score
        compatibility_score = min(100, int((riasec_match_score * 15) + (personality_match_score * 10) + 50))
        
        return {
            "primary_match": primary_match or top_interests[0].title(),
            "secondary_match": secondary_match or (top_interests[1].title() if len(top_interests) > 1 else None),
            "compatibility_score": compatibility_score
        }
        
    except Exception as e:
        print(f"❌ Error calculating personality match: {str(e)}")
        return {
            "primary_match": "General",
            "secondary_match": None,
            "compatibility_score": 50
        }

# ====================================================================
# CAREER SEARCH FUNCTION
# ====================================================================

def search_personality_careers(big5_scores: Dict[str, float], 
                              riasec_scores: Dict[str, float],
                              top_k: int = 20) -> List[Dict[str, Any]]:
    """Search careers based on personality scores"""
    try:
        start_time = time.time()
        
        # Convert scores to search query
        search_query = convert_scores_to_query(big5_scores, riasec_scores)
        print(f"🔍 Search query: '{search_query}'")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([search_query]).astype('float32')
        
        # Search FAISS index
        distances, indices = faiss_index.search(query_embedding, min(top_k * 2, len(careers_df)))
        
        results = []
        seen_careers = set()
        
        for distance, idx in zip(distances[0], indices[0]):
            career = careers_df.iloc[idx]
            
            # Avoid duplicates
            career_id = str(career['career_id'])
            if career_id in seen_careers:
                continue
            seen_careers.add(career_id)
            
            # Calculate personality match
            personality_fit = calculate_personality_match(career, big5_scores, riasec_scores)
            
            # Calculate combined match score (70% semantic + 30% personality)
            semantic_score = max(0, int((1 / (1 + distance)) * 100))
            combined_score = int(semantic_score * 0.7 + personality_fit["compatibility_score"] * 0.3)
            
            # Format salary
            salary = career.get('avg_salary', 0)
            if salary and salary > 0:
                if salary >= 100000:
                    salary_range = f"₹{salary//100000}L - ₹{int(salary*1.3)//100000}L"
                else:
                    salary_range = f"₹{salary:,} - ₹{int(salary*1.3):,}"
            else:
                salary_range = "Not specified"
            
            # Extract key skills
            skills_text = str(career.get('skills', ''))
            key_skills = [skill.strip() for skill in skills_text.split(',')[:6] if skill.strip()]
            
            # Build result
            result = {
                "id": f"career_{career_id}",
                "title": str(career.get('name', 'Unknown Career')),
                "summary": str(career.get('description', 'No description available'))[:150] + "...",
                "match_score": combined_score,
                "salary_range": salary_range,
                "education": str(career.get('education', 'Not specified')),
                "industry": str(career.get('industry', 'General')),
                "work_environment": "Office/Remote/Hybrid",
                "key_skills": key_skills,
                "personality_fit": personality_fit,
                "riasec_match": str(career.get('riasec', 'General')),
                "location_options": str(career.get('location_options', 'Multiple locations'))
            }
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return results, search_query, processing_time
        
    except Exception as e:
        print(f"❌ Error in personality career search: {str(e)}")
        raise

# ====================================================================
# STARTUP EVENT
# ====================================================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS etc.
    allow_headers=["*"],  # allow all headers
)


@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global supabase_client, careers_df, embedding_model, faiss_index, career_embeddings
    
    try:
        print("🚀 Starting up Career Recommender API...")
        
        # Initialize Supabase
        supabase_client = setup_supabase_connection()
        
        # Load career data
        careers_df = load_careers_data(supabase_client)
        
        # Initialize embedding model
        embedding_model = initialize_embedding_model()
        
        # Generate embeddings
        career_embeddings = generate_career_embeddings(careers_df, embedding_model)
        
        # Build FAISS index
        faiss_index = build_faiss_index(career_embeddings)
        
        print("✅ All components initialized successfully!")
        
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")
        raise

# ====================================================================
# API ENDPOINTS
# ====================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "success",
        "message": "Personality Career Recommender API is running!",
        "version": "1.0.0",
        "careers_loaded": len(careers_df) if careers_df is not None else 0
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "supabase": supabase_client is not None,
            "careers_data": careers_df is not None,
            "embedding_model": embedding_model is not None,
            "faiss_index": faiss_index is not None
        },
        "careers_count": len(careers_df) if careers_df is not None else 0
    }

@app.post("/personality-career-match", response_model=PersonalityResponse)
async def get_personality_career_matches(request: PersonalityRequest):
    """Main endpoint for personality-based career matching"""
    try:
        # Check if system is ready
        if not all([supabase_client, careers_df is not None, embedding_model, faiss_index]):
            raise HTTPException(status_code=503, detail="System not ready. Please try again in a moment.")
        
        # Convert pydantic models to dicts
        big5_dict = request.big5_scores.dict()
        riasec_dict = request.riasec_scores.dict()
        
        # Perform search
        results, query_processed, processing_time = search_personality_careers(
            big5_dict, riasec_dict, top_k=20
        )
        
        # Generate personality profile summary
        top_big5 = sorted(big5_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        top_riasec = sorted(riasec_dict.items(), key=lambda x: x[1], reverse=True)[:2]
        
        personality_profile = {
            "dominant_interests": [item[0].title() for item in top_riasec],
            "work_style": [f"{item[0].replace('_', ' ').title()}" for item in top_big5 if item[1] >= SCORE_THRESHOLDS["high"]]
        }
        
        # Format response
        response = PersonalityResponse(
            status="success",
            query_processed=query_processed,
            total_results=len(results),
            processing_time_ms=processing_time,
            personality_profile=personality_profile,
            results=results,
            filters_applied={
                "personality_focus": "_".join([item[0] for item in top_riasec]),
                "work_style": "_".join([item[0] for item in top_big5[:2]]),
                "excluded_low_matches": True
            }
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error in career matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ====================================================================
# RUN SERVER
# ====================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )