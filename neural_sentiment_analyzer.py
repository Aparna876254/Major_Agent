"""
Neural Network Sentiment Analyzer for E-Commerce Price Comparison
Uses DistilBERT fine-tuned on SST-2 for sentiment analysis
Can be further fine-tuned on Amazon/Yelp review datasets

Datasets:
- Amazon Polarity: https://huggingface.co/datasets/mteb/amazon_polarity
- Amazon Reviews 2023: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Yelp Reviews: https://huggingface.co/datasets/Yelp/yelp_review_full

Model:
- DistilBERT SST-2: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
"""

import os
import re
import warnings
from typing import Any, cast
warnings.filterwarnings('ignore')

# Check for transformers availability
try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers library not installed. Run: pip install transformers torch")


class NeuralSentimentAnalyzer:
    """
    Neural network-based sentiment analyzer using DistilBERT.
    Pre-trained on SST-2 and can be fine-tuned on Amazon/Yelp datasets.
    """
    
    def __init__(self, model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the neural sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name/path
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the pre-trained DistilBERT model"""
        try:
            print(f"🧠 Loading Neural Sentiment Model...")
            print(f"   Model: {self.model_name}")
            print(f"   Device: {self.device}")
            
            # Use pipeline for easy inference
            self.pipeline = cast(Any, hf_pipeline)(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=512
            )
            
            self.is_ready = True
            print(f"✅ Neural Sentiment Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_ready = False
    
    def preprocess_text(self, text):
        """Clean and prepare text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Truncate very long texts (DistilBERT has 512 token limit)
        if len(text) > 2000:
            text = text[:2000]
        
        return text.strip()
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text.
        
        Returns:
            dict: {
                'sentiment': 'positive'/'negative'/'neutral',
                'score': float (0-1),
                'confidence': dict with positive/negative scores,
                'label': original model label
            }
        """
        if not self.is_ready or not text:
            return {
                'sentiment': 'unknown',
                'score': 0.5,
                'confidence': {'positive': 0.5, 'negative': 0.5},
                'label': 'UNKNOWN'
            }
        
        try:
            clean_text = self.preprocess_text(text)
            if not clean_text:
                return {
                    'sentiment': 'unknown',
                    'score': 0.5,
                    'confidence': {'positive': 0.5, 'negative': 0.5},
                    'label': 'UNKNOWN'
                }
            
            # Get prediction from model
            if self.pipeline is None:
                return {
                    'sentiment': 'unknown',
                    'score': 0.5,
                    'confidence': {'positive': 0.5, 'negative': 0.5},
                    'label': 'UNKNOWN'
                }
            result = self.pipeline(clean_text)[0]
            
            label = result['label']  # POSITIVE or NEGATIVE
            score = result['score']  # Confidence score
            
            # Map to our sentiment categories
            if label == 'POSITIVE':
                sentiment = 'positive' if score > 0.6 else 'neutral'
                confidence = {'positive': score, 'negative': 1 - score}
            else:  # NEGATIVE
                sentiment = 'negative' if score > 0.6 else 'neutral'
                confidence = {'positive': 1 - score, 'negative': score}
            
            # Determine final score (0 = negative, 0.5 = neutral, 1 = positive)
            if sentiment == 'positive':
                final_score = 0.5 + (score * 0.5)
            elif sentiment == 'negative':
                final_score = 0.5 - (score * 0.5)
            else:
                final_score = 0.5
            
            return {
                'sentiment': sentiment,
                'score': final_score,
                'confidence': confidence,
                'label': label
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing text: {e}")
            return {
                'sentiment': 'unknown',
                'score': 0.5,
                'confidence': {'positive': 0.5, 'negative': 0.5},
                'label': 'ERROR'
            }
    
    def analyze_product(self, product_dict):
        """
        Analyze sentiment for a product based on its reviews and description.
        
        Args:
            product_dict: Product data dictionary with 'customer_reviews', 'description', etc.
        
        Returns:
            Updated product_dict with sentiment fields added
        """
        if not self.is_ready:
            product_dict['sentiment'] = 'unknown'
            product_dict['sentiment_score'] = 0.5
            product_dict['sentiment_emoji'] = '❓'
            product_dict['sentiment_source'] = 'none'
            return product_dict
        
        # Collect text to analyze
        texts_to_analyze = []
        source = 'description'
        
        # Priority 1: Customer reviews (most reliable for sentiment)
        if product_dict.get('customer_reviews'):
            reviews = product_dict['customer_reviews']
            for review in reviews[:5]:  # Analyze up to 5 reviews
                if isinstance(review, dict):
                    review_text = review.get('text', '') or review.get('title', '')
                else:
                    review_text = str(review)
                if review_text and len(review_text) > 10:
                    texts_to_analyze.append(review_text)
            if texts_to_analyze:
                source = 'customer_reviews'
        
        # Priority 2: Review summary
        if not texts_to_analyze and product_dict.get('review_summary'):
            texts_to_analyze.append(product_dict['review_summary'])
            source = 'review_summary'
        
        # Priority 3: Product description/features
        if not texts_to_analyze:
            if product_dict.get('description'):
                texts_to_analyze.append(product_dict['description'])
            if product_dict.get('features'):
                if isinstance(product_dict['features'], list):
                    texts_to_analyze.extend(product_dict['features'][:3])
                else:
                    texts_to_analyze.append(str(product_dict['features']))
            source = 'description'
        
        # Analyze collected texts
        if not texts_to_analyze:
            product_dict['sentiment'] = 'unknown'
            product_dict['sentiment_score'] = 0.5
            product_dict['sentiment_emoji'] = '❓'
            product_dict['sentiment_source'] = 'none'
            return product_dict
        
        # Aggregate sentiment from multiple texts
        sentiments = []
        scores = []
        
        for text in texts_to_analyze:
            result = self.analyze_text(text)
            sentiments.append(result['sentiment'])
            scores.append(result['score'])
        
        # Calculate aggregate sentiment
        avg_score = sum(scores) / len(scores)
        
        # Determine final sentiment based on average score
        if avg_score >= 0.6:
            final_sentiment = 'positive'
        elif avg_score <= 0.4:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        # Generate explanation
        pos_count = sentiments.count('positive')
        neg_count = sentiments.count('negative')
        neu_count = sentiments.count('neutral')
        
        if source == 'customer_reviews':
            explanation = f"Based on {len(texts_to_analyze)} reviews: {pos_count} positive, {neg_count} negative, {neu_count} neutral"
        else:
            explanation = f"Analyzed from product {source}"
        
        # Update product dictionary
        product_dict['sentiment'] = final_sentiment
        product_dict['sentiment_score'] = avg_score
        product_dict['sentiment_emoji'] = self.get_sentiment_emoji(final_sentiment)
        product_dict['sentiment_source'] = source
        product_dict['sentiment_explanation'] = explanation
        product_dict['sentiment_confidence'] = {
            'positive': pos_count / len(sentiments) if sentiments else 0,
            'neutral': neu_count / len(sentiments) if sentiments else 0,
            'negative': neg_count / len(sentiments) if sentiments else 0
        }
        
        return product_dict
    
    def analyze_products_batch(self, products_list):
        """
        Analyze sentiment for multiple products.
        
        Args:
            products_list: List of product dictionaries
            
        Returns:
            List of products with sentiment fields added
        """
        if not self.is_ready:
            for product in products_list:
                product['sentiment'] = 'unknown'
                product['sentiment_score'] = 0.5
                product['sentiment_emoji'] = '❓'
            return products_list
        
        print(f"🧠 Analyzing sentiment for {len(products_list)} products using Neural Network...")
        
        analyzed = []
        for i, product in enumerate(products_list):
            try:
                self.analyze_product(product)
                analyzed.append(product)
            except Exception as e:
                product['sentiment'] = 'unknown'
                product['sentiment_score'] = 0.5
                product['sentiment_emoji'] = '❓'
                analyzed.append(product)
        
        # Print summary
        sentiments = [p.get('sentiment', 'unknown') for p in analyzed]
        pos = sentiments.count('positive')
        neu = sentiments.count('neutral')
        neg = sentiments.count('negative')
        print(f"   📊 Neural Sentiment Analysis: 😊 {pos} positive, 😐 {neu} neutral, 😞 {neg} negative")
        
        return analyzed
    
    @staticmethod
    def get_sentiment_emoji(sentiment):
        """Get emoji for sentiment label"""
        emoji_map = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞',
            'unknown': '❓'
        }
        return emoji_map.get(sentiment, '❓')


class DatasetLoader:
    """
    Utility class for loading and preparing datasets for fine-tuning.
    Supports Amazon Polarity, Amazon Reviews 2023, and Yelp Reviews.
    """
    
    @staticmethod
    def load_amazon_polarity(sample_size=None):
        """
        Load Amazon Polarity dataset from HuggingFace.
        
        Dataset: https://huggingface.co/datasets/mteb/amazon_polarity
        Labels: 0 (negative), 1 (positive)
        """
        try:
            from datasets import load_dataset
            
            print("📚 Loading Amazon Polarity dataset...")
            dataset = load_dataset("mteb/amazon_polarity", split="test")
            
            if sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
            
            print(f"   Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading Amazon Polarity: {e}")
            return None
    
    @staticmethod
    def load_amazon_reviews_2023(category="Electronics", sample_size=None):
        """
        Load Amazon Reviews 2023 dataset from HuggingFace.
        
        Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
        Categories: Electronics, Books, Clothing_Shoes_and_Jewelry, etc.
        """
        try:
            from datasets import load_dataset
            
            print(f"📚 Loading Amazon Reviews 2023 ({category})...")
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                split="full",
                trust_remote_code=True
            )
            
            if sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
            
            print(f"   Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading Amazon Reviews 2023: {e}")
            return None
    
    @staticmethod
    def load_yelp_reviews(sample_size=None):
        """
        Load Yelp Review Full dataset from HuggingFace.
        
        Dataset: https://huggingface.co/datasets/Yelp/yelp_review_full
        Labels: 1-5 stars
        """
        try:
            from datasets import load_dataset
            
            print("📚 Loading Yelp Reviews dataset...")
            dataset = load_dataset("Yelp/yelp_review_full", split="test")
            
            if sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
            
            print(f"   Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading Yelp Reviews: {e}")
            return None
    
    @staticmethod
    def prepare_combined_dataset(amazon_samples=5000, yelp_samples=5000):
        """
        Prepare a combined dataset from multiple sources for fine-tuning.
        
        Returns:
            List of dicts with 'text' and 'label' (0=negative, 1=neutral, 2=positive)
        """
        combined_data = []
        
        # Load Amazon Polarity
        amazon_data = DatasetLoader.load_amazon_polarity(amazon_samples)
        if amazon_data:
            for item in amazon_data:
                # Amazon polarity: 0=negative, 1=positive
                item_dict = cast(dict[str, Any], item)
                label = 2 if item_dict['label'] == 1 else 0  # Map to our 3-class
                combined_data.append({
                    'text': item_dict['text'],
                    'label': label,
                    'source': 'amazon_polarity'
                })
        
        # Load Yelp Reviews
        yelp_data = DatasetLoader.load_yelp_reviews(yelp_samples)
        if yelp_data:
            for item in yelp_data:
                # Yelp: 1-5 stars -> 0 (1-2), 1 (3), 2 (4-5)
                item_dict = cast(dict[str, Any], item)
                stars = item_dict['label'] + 1  # Dataset is 0-indexed
                if stars <= 2:
                    label = 0  # Negative
                elif stars == 3:
                    label = 1  # Neutral
                else:
                    label = 2  # Positive
                combined_data.append({
                    'text': item_dict['text'],
                    'label': label,
                    'source': 'yelp'
                })
        
        print(f"✅ Combined dataset: {len(combined_data)} samples")
        return combined_data


# Convenience function for easy import
def get_neural_analyzer():
    """Get a ready-to-use neural sentiment analyzer instance"""
    return NeuralSentimentAnalyzer()


# Test the analyzer if run directly
if __name__ == "__main__":
    print("=" * 70)
    print("Neural Sentiment Analyzer - Test")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n❌ Please install required packages:")
        print("   pip install transformers torch datasets")
        exit(1)
    
    # Initialize analyzer
    analyzer = NeuralSentimentAnalyzer()
    
    if analyzer.is_ready:
        # Test with sample texts
        test_texts = [
            "This product is absolutely amazing! Best purchase I've ever made.",
            "Terrible quality, broke after one day. Complete waste of money.",
            "It's okay, nothing special. Does what it's supposed to do.",
            "The phone has great camera but battery life is disappointing.",
            "Excellent value for money, highly recommend!"
        ]
        
        print("\n📝 Testing Neural Sentiment Analysis:\n")
        for text in test_texts:
            result = analyzer.analyze_text(text)
            emoji = analyzer.get_sentiment_emoji(result['sentiment'])
            print(f"{emoji} [{result['sentiment'].upper():8}] (Score: {result['score']:.2f})")
            print(f"   \"{text[:60]}...\"" if len(text) > 60 else f"   \"{text}\"")
            print()
        
        # Test with mock product
        print("\n📦 Testing Product Analysis:\n")
        mock_product = {
            'name': 'Samsung Galaxy S24',
            'customer_reviews': [
                {'text': 'Amazing phone, camera is incredible!'},
                {'text': 'Battery could be better but overall great'},
                {'text': 'Best phone I have ever owned'}
            ]
        }
        
        analyzer.analyze_product(mock_product)
        print(f"Product: {mock_product['name']}")
        print(f"Sentiment: {mock_product['sentiment_emoji']} {mock_product['sentiment'].upper()}")
        print(f"Score: {mock_product['sentiment_score']:.2f}")
        print(f"Source: {mock_product['sentiment_source']}")
        print(f"Explanation: {mock_product['sentiment_explanation']}")
