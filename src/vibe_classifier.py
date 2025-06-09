import json
import logging
import re
from typing import List, Dict, Any
from transformers import pipeline

logger = logging.getLogger(__name__)

class VibeClassifier:
    # Define supported vibes and their keywords
    SUPPORTED_VIBES = {
        "Coquette": {
            "keywords": ["coquette", "romantic", "flirty", "feminine", "delicate", "lace", "ribbon", "bow", "pink", "pastel", "linen", "summer", "easy-breezy"],
            "hashtags": ["#coquette", "#coquetteaesthetic", "#coquettefashion", "#coquettestyle", "#linnenset", "#summeroutfit"]
        },
        "Clean Girl": {
            "keywords": ["clean", "minimal", "simple", "neutral", "basic", "elegant", "sophisticated", "classic", "cotton", "linen"],
            "hashtags": ["#cleangirl", "#cleangirlaesthetic", "#minimalist", "#minimalstyle", "#linnenset"]
        },
        "Cottagecore": {
            "keywords": ["cottage", "vintage", "rustic", "floral", "nature", "garden", "romantic", "whimsical", "linen", "summer"],
            "hashtags": ["#cottagecore", "#cottagecoreaesthetic", "#cottagecorefashion", "#linnenset"]
        },
        "Streetcore": {
            "keywords": ["street", "urban", "edgy", "cool", "casual", "sporty", "streetwear", "streetstyle"],
            "hashtags": ["#streetcore", "#streetstyle", "#streetwear", "#urbanstyle"]
        },
        "Y2K": {
            "keywords": ["y2k", "retro", "vintage", "90s", "2000s", "nostalgic", "playful", "fun"],
            "hashtags": ["#y2k", "#y2kaesthetic", "#y2kfashion", "#y2kstyle"]
        },
        "Boho": {
            "keywords": ["boho", "bohemian", "hippie", "free", "spirit", "natural", "flowy", "ethnic", "linen"],
            "hashtags": ["#boho", "#bohostyle", "#bohemian", "#bohochic", "#linnenset"]
        },
        "Party Glam": {
            "keywords": ["party", "glam", "glamorous", "sparkle", "shiny", "dressy", "elegant", "fancy"],
            "hashtags": ["#partylook", "#glam", "#glamorous", "#partyoutfit"]
        }
    }

    def __init__(self, vibes_list_path: str, debug: bool = False):
        """Initialize the vibe classifier with rule-based and ML components."""
        self.debug = debug
        try:
            # Load vibes list for validation
            with open(vibes_list_path, 'r') as f:
                self.vibes_list = json.load(f)
            
            # Initialize sentiment analyzer for context
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            
            if self.debug:
                logger.info("VibeClassifier initialized with debug mode")
                
        except Exception as e:
            logger.error(f"Error initializing VibeClassifier: {str(e)}")
            raise

    def _calculate_vibe_score(self, text: str, vibe: str) -> float:
        """Calculate a score for a vibe based on keyword matches and context."""
        score = 0.0
        vibe_info = self.SUPPORTED_VIBES[vibe]
        
        # Check for keyword matches
        for keyword in vibe_info["keywords"]:
            if keyword.lower() in text.lower():
                score += 0.2  # Increased from 0.1
                
        # Check for hashtag matches
        for hashtag in vibe_info["hashtags"]:
            if hashtag.lower() in text.lower():
                score += 0.3  # Increased from 0.2
                
        # Get sentiment for context
        try:
            sentiment = self.sentiment_analyzer(text)[0]
            if sentiment['label'] == 'POSITIVE':
                score += 0.2  # Increased from 0.1
        except:
            pass
            
        return min(score, 1.0)  # Cap score at 1.0

    def process_caption(self, caption: str = None, hashtags: List[str] = None) -> List[Dict[str, Any]]:
        """Process caption and hashtags to classify vibes."""
        try:
            if not caption and not hashtags:
                if self.debug:
                    logger.warning("No caption or hashtags provided")
                return []
                
            # Combine caption and hashtags
            text = ""
            if caption:
                text += caption + " "
            if hashtags:
                text += " ".join(hashtags)
                
            if self.debug:
                logger.info(f"Processing text: {text}")
            
            # Calculate scores for each vibe
            vibe_scores = []
            for vibe in self.SUPPORTED_VIBES:
                score = self._calculate_vibe_score(text, vibe)
                if score > 0.2:  # Lowered threshold from 0.3
                    vibe_scores.append({
                        "vibe": vibe,
                        "confidence": score
                    })
            
            # Sort by confidence and take top 3
            vibe_scores.sort(key=lambda x: x["confidence"], reverse=True)
            top_vibes = vibe_scores[:3]
            
            if self.debug:
                logger.info(f"Detected vibes: {top_vibes}")
                
            return top_vibes
            
        except Exception as e:
            logger.error(f"Error processing caption: {str(e)}")
            return [] 
