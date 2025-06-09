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
            "keywords": ["coquette", "romantic", "flirty", "feminine", "delicate", "lace", "ribbon", "bow", "pink", "pastel"],
            "hashtags": ["#coquette", "#coquetteaesthetic", "#coquettefashion", "#coquettestyle"],
            "product_attributes": {
                "colors": ["pink", "pastel", "white", "cream", "beige"],
                "types": ["dress", "skirt", "blouse", "vest", "trousers"],
                "styles": ["feminine", "delicate", "romantic"]
            }
        },
        "Clean Girl": {
            "keywords": ["clean", "minimal", "simple", "neutral", "basic", "elegant", "sophisticated", "classic"],
            "hashtags": ["#cleangirl", "#cleangirlaesthetic", "#minimalist", "#minimalstyle"],
            "product_attributes": {
                "colors": ["white", "black", "beige", "cream", "neutral"],
                "types": ["vest", "trousers", "shirt", "dress", "jacket"],
                "styles": ["minimal", "simple", "elegant"]
            }
        },
        "Cottagecore": {
            "keywords": ["cottage", "vintage", "rustic", "floral", "nature", "garden", "romantic", "whimsical"],
            "hashtags": ["#cottagecore", "#cottagecoreaesthetic", "#cottagecorefashion"],
            "product_attributes": {
                "colors": ["floral", "pastel", "earth", "green", "brown"],
                "types": ["dress", "skirt", "blouse", "vest"],
                "styles": ["vintage", "romantic", "whimsical"]
            }
        },
        "Streetcore": {
            "keywords": ["street", "urban", "edgy", "cool", "casual", "sporty", "streetwear", "streetstyle"],
            "hashtags": ["#streetcore", "#streetstyle", "#streetwear", "#urbanstyle"],
            "product_attributes": {
                "colors": ["black", "white", "gray", "navy", "red"],
                "types": ["trousers", "jacket", "hoodie", "sneakers"],
                "styles": ["casual", "sporty", "urban"]
            }
        },
        "Y2K": {
            "keywords": ["y2k", "retro", "vintage", "90s", "2000s", "nostalgic", "playful", "fun"],
            "hashtags": ["#y2k", "#y2kaesthetic", "#y2kfashion", "#y2kstyle"],
            "product_attributes": {
                "colors": ["pink", "purple", "blue", "silver", "metallic"],
                "types": ["trousers", "top", "dress", "skirt"],
                "styles": ["retro", "playful", "fun"]
            }
        },
        "Boho": {
            "keywords": ["boho", "bohemian", "hippie", "free", "spirit", "natural", "flowy", "ethnic"],
            "hashtags": ["#boho", "#bohostyle", "#bohemian", "#bohochic"],
            "product_attributes": {
                "colors": ["earth", "natural", "ethnic", "multicolor"],
                "types": ["dress", "skirt", "vest", "blouse"],
                "styles": ["flowy", "natural", "ethnic"]
            }
        },
        "Party Glam": {
            "keywords": ["party", "glam", "glamorous", "sparkle", "shiny", "dressy", "elegant", "fancy"],
            "hashtags": ["#partylook", "#glam", "#glamorous", "#partyoutfit"],
            "product_attributes": {
                "colors": ["black", "white", "red", "gold", "silver"],
                "types": ["dress", "vest", "trousers", "top"],
                "styles": ["elegant", "glamorous", "dressy"]
            }
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
                score += 0.2
                
        # Check for hashtag matches
        for hashtag in vibe_info["hashtags"]:
            if hashtag.lower() in text.lower():
                score += 0.3
                
        # Get sentiment for context
        try:
            sentiment = self.sentiment_analyzer(text)[0]
            if sentiment['label'] == 'POSITIVE':
                score += 0.2
        except:
            pass
            
        return min(score, 1.0)

    def _calculate_product_vibe_score(self, product: Dict[str, Any], vibe: str) -> float:
        """Calculate a score for a vibe based on product attributes."""
        score = 0.0
        vibe_info = self.SUPPORTED_VIBES[vibe]
        product_attrs = vibe_info["product_attributes"]
        
        # Check color match
        product_color = product.get("color", "").lower()
        for color in product_attrs["colors"]:
            if color.lower() in product_color:
                score += 0.3
                break
                
        # Check type match
        product_type = product.get("type", "").lower()
        for type_ in product_attrs["types"]:
            if type_.lower() in product_type:
                score += 0.3
                break
                
        # Check style match (if available)
        product_style = product.get("style", "").lower()
        if product_style:
            for style in product_attrs["styles"]:
                if style.lower() in product_style:
                    score += 0.2
                    break
                    
        return min(score, 1.0)

    def process_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process products to classify vibes."""
        try:
            if not products:
                if self.debug:
                    logger.warning("No products provided")
                return []
                
            # Calculate scores for each vibe based on all products
            vibe_scores = {}
            for product in products:
                for vibe in self.SUPPORTED_VIBES:
                    score = self._calculate_product_vibe_score(product, vibe)
                    vibe_scores[vibe] = vibe_scores.get(vibe, 0) + score
                    
            # Normalize scores by number of products
            num_products = len(products)
            vibe_scores = {vibe: score/num_products for vibe, score in vibe_scores.items()}
            
            # Convert to list and filter by threshold
            vibe_scores_list = [
                {"vibe": vibe, "confidence": score}
                for vibe, score in vibe_scores.items()
                if score > 0.2
            ]
            
            # Sort by confidence
            vibe_scores_list.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Ensure we return at least 1 and at most 3 vibes
            if not vibe_scores_list:
                # If no vibes meet threshold, return the top vibe anyway
                top_vibe = max(vibe_scores.items(), key=lambda x: x[1])[0]
                vibe_scores_list = [{
                    "vibe": top_vibe,
                    "confidence": vibe_scores[top_vibe]
                }]
            
            # Take top 3 vibes
            top_vibes = vibe_scores_list[:3]
            
            if self.debug:
                logger.info(f"Detected vibes from products: {top_vibes}")
                
            return top_vibes
            
        except Exception as e:
            logger.error(f"Error processing products: {str(e)}")
            return []

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
                if score > 0.2:  # Lowered threshold to ensure we get at least one vibe
                    vibe_scores.append({
                        "vibe": vibe,
                        "confidence": score
                    })
            
            # Sort by confidence
            vibe_scores.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Ensure we return at least 1 and at most 3 vibes
            if not vibe_scores:
                # If no vibes meet threshold, return the top vibe anyway
                top_vibe = max(self.SUPPORTED_VIBES.keys(), 
                             key=lambda v: self._calculate_vibe_score(text, v))
                vibe_scores = [{
                    "vibe": top_vibe,
                    "confidence": self._calculate_vibe_score(text, top_vibe)
                }]
            
            # Take top 3 vibes
            top_vibes = vibe_scores[:3]
            
            if self.debug:
                logger.info(f"Detected vibes: {top_vibes}")
                
            return top_vibes
            
        except Exception as e:
            logger.error(f"Error processing caption: {str(e)}")
            return [] 
