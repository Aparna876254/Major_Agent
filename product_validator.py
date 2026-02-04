"""
Product Validation System for E-Commerce Scraping
Verifies that scraped products match the search query accurately
Handles: Brand + Series + Model Number verification

Author: Major Project - E-Commerce Price Comparison
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ProductValidation:
    """Result of product validation"""
    is_valid: bool
    brand_match: bool
    series_match: bool
    model_match: bool
    confidence: float
    reason: str


class ProductValidator:
    """
    Validates scraped products against search queries.
    Ensures iPhone 16 search doesn't return iPhone 17, etc.
    
    Logic:
    - Brand + Product Line: No strict verification (Apple + iPhone is OK)
    - Model/Series/Number: MUST be verified strictly
    """
    
    # Brand configurations with their product lines and series
    BRAND_CONFIG = {
        # Smartphones
        'apple': {
            'product_lines': ['iphone', 'ipad', 'macbook', 'mac', 'watch', 'airpods', 'homepod'],
            'series': {
                'iphone': ['pro', 'pro max', 'plus', 'mini', 'se'],
                'ipad': ['pro', 'air', 'mini'],
                'macbook': ['pro', 'air'],
                'watch': ['ultra', 'se', 'series']
            },
            'model_pattern': r'\b(\d{1,2})\b'  # iPhone 15, 16, etc.
        },
        'samsung': {
            'product_lines': ['galaxy', 'tab', 'buds', 'watch', 'tv'],
            'series': {
                'galaxy': ['s', 'a', 'm', 'f', 'z', 'fold', 'flip', 'note'],
                'tab': ['s', 'a'],
                'watch': ['ultra', 'classic']
            },
            'model_pattern': r'\b([a-z])?\s*(\d{1,2})\b'  # S24, A54, M34
        },
        'vivo': {
            'product_lines': ['vivo'],
            'series': {
                'vivo': ['v', 'y', 't', 'x', 's', 'iqoo']
            },
            'model_pattern': r'\b([a-z])\s*(\d{1,3})\b'  # V30, Y100, T3
        },
        'oppo': {
            'product_lines': ['oppo', 'reno', 'find'],
            'series': {
                'oppo': ['reno', 'f', 'a', 'find', 'k'],
                'reno': ['pro', 'plus'],
                'find': ['x', 'n']
            },
            'model_pattern': r'\b(\d{1,2})\b'  # Reno 11, Find X7
        },
        'xiaomi': {
            'product_lines': ['xiaomi', 'redmi', 'poco', 'mi'],
            'series': {
                'redmi': ['note', 'a', 'k', 'pro'],
                'poco': ['x', 'f', 'm', 'c'],
                'xiaomi': ['ultra', 'pro', 'lite']
            },
            'model_pattern': r'\b(\d{1,2})\s*(pro|ultra|lite|plus)?\b'
        },
        'realme': {
            'product_lines': ['realme'],
            'series': {
                'realme': ['gt', 'narzo', 'c', 'p', 'number']
            },
            'model_pattern': r'\b(\d{1,2})\s*(pro|plus|x)?\b'
        },
        'oneplus': {
            'product_lines': ['oneplus'],
            'series': {
                'oneplus': ['nord', 'open', 'ace', 'number']
            },
            'model_pattern': r'\b(\d{1,2})\s*(t|r|pro)?\b'  # 12, 12R, Nord CE3
        },
        'motorola': {
            'product_lines': ['motorola', 'moto'],
            'series': {
                'moto': ['g', 'e', 'edge', 'razr']
            },
            'model_pattern': r'\b([a-z])\s*(\d{1,2})\b'
        },
        'nothing': {
            'product_lines': ['nothing', 'phone'],
            'series': {
                'phone': ['1', '2', '2a']
            },
            'model_pattern': r'\b(\d)\s*(a)?\b'
        },
        'google': {
            'product_lines': ['pixel', 'nest'],
            'series': {
                'pixel': ['pro', 'a', 'fold']
            },
            'model_pattern': r'\b(\d)\s*(a|pro|xl)?\b'
        },
        # Laptops
        'dell': {
            'product_lines': ['inspiron', 'xps', 'latitude', 'vostro', 'alienware', 'precision'],
            'series': {
                'inspiron': ['14', '15', '16', '13'],
                'xps': ['13', '15', '17']
            },
            'model_pattern': r'\b(\d{2,4})\b'
        },
        'hp': {
            'product_lines': ['pavilion', 'envy', 'spectre', 'omen', 'victus', 'elitebook', 'probook'],
            'series': {
                'pavilion': ['x360', 'gaming'],
                'omen': ['15', '16', '17']
            },
            'model_pattern': r'\b(\d{2,4})\b'
        },
        'lenovo': {
            'product_lines': ['thinkpad', 'ideapad', 'legion', 'yoga', 'thinkbook', 'v'],
            'series': {
                'thinkpad': ['x', 't', 'e', 'l', 'p'],
                'legion': ['5', '7', 'pro', 'slim']
            },
            'model_pattern': r'\b(\d{1,4})\b'
        },
        'asus': {
            'product_lines': ['vivobook', 'zenbook', 'rog', 'tuf', 'proart', 'expertbook'],
            'series': {
                'rog': ['strix', 'zephyrus', 'flow'],
                'tuf': ['gaming', 'dash']
            },
            'model_pattern': r'\b(\d{1,4})\b'
        },
        'acer': {
            'product_lines': ['aspire', 'swift', 'nitro', 'predator', 'spin'],
            'series': {
                'aspire': ['1', '3', '5', '7'],
                'nitro': ['5', '7']
            },
            'model_pattern': r'\b(\d)\b'
        },
        # TVs
        'lg': {
            'product_lines': ['oled', 'nanocell', 'qned', 'uhd'],
            'series': {
                'oled': ['c', 'g', 'b', 'z']
            },
            'model_pattern': r'\b([a-z]?\d{1,2})\b'
        },
        'sony': {
            'product_lines': ['bravia', 'alpha', 'wh', 'wf'],
            'series': {
                'bravia': ['a', 'x'],
                'alpha': ['7', '9', '1']
            },
            'model_pattern': r'\b([a-z]?\d{1,4})\b'
        },
        # Audio
        'bose': {
            'product_lines': ['quietcomfort', 'soundlink', 'sleepbuds'],
            'series': {},
            'model_pattern': r'\b(\d{2,3})\b'
        },
        'jbl': {
            'product_lines': ['tune', 'flip', 'charge', 'partybox', 'go', 'xtreme'],
            'series': {},
            'model_pattern': r'\b(\d{1,3})\b'
        },
        'sennheiser': {
            'product_lines': ['hd', 'momentum', 'ie'],
            'series': {},
            'model_pattern': r'\b(\d{2,4})\b'
        },
        # Gaming
        'sony_gaming': {
            'product_lines': ['playstation', 'ps', 'dualsense'],
            'series': {},
            'model_pattern': r'\b(\d)\b'  # PS5, PS4
        },
        'microsoft_gaming': {
            'product_lines': ['xbox', 'series'],
            'series': {
                'xbox': ['series x', 'series s', 'one']
            },
            'model_pattern': r'(series\s*[xs]|one)'
        },
        'nintendo': {
            'product_lines': ['switch', 'ds'],
            'series': {
                'switch': ['lite', 'oled']
            },
            'model_pattern': r'(lite|oled)?'
        },
        # Cameras
        'canon': {
            'product_lines': ['eos', 'powershot', 'rebel'],
            'series': {
                'eos': ['r', 'm', 'd']
            },
            'model_pattern': r'\b(\d{1,4}d?)\b'
        },
        'nikon': {
            'product_lines': ['d', 'z', 'coolpix'],
            'series': {},
            'model_pattern': r'\b([dz]?\d{1,4})\b'
        },
        # Wearables
        'fitbit': {
            'product_lines': ['versa', 'sense', 'charge', 'inspire', 'luxe'],
            'series': {},
            'model_pattern': r'\b(\d)\b'
        },
        'garmin': {
            'product_lines': ['forerunner', 'fenix', 'venu', 'vivoactive'],
            'series': {},
            'model_pattern': r'\b(\d{2,4})\b'
        },
        'boat': {
            'product_lines': ['airdopes', 'rockerz', 'stone', 'bassheads'],
            'series': {},
            'model_pattern': r'\b(\d{2,3})\b'
        },
        'noise': {
            'product_lines': ['colorfit', 'buds', 'shots'],
            'series': {
                'colorfit': ['pro', 'pulse', 'icon', 'ultra']
            },
            'model_pattern': r'\b(\d{1,2})\b'
        },
        'amazfit': {
            'product_lines': ['gtr', 'gts', 'bip', 't-rex', 'balance'],
            'series': {},
            'model_pattern': r'\b(\d)\b'
        }
    }
    
    def __init__(self):
        """Initialize the product validator"""
        self.validation_stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'brand_mismatch': 0,
            'series_mismatch': 0,
            'model_mismatch': 0
        }
    
    def parse_search_query(self, query: str) -> Dict:
        """
        Parse search query to extract brand, product line, series, and model number.
        
        Args:
            query: Search query like "iPhone 16 Pro Max" or "Samsung Galaxy S24 Ultra"
            
        Returns:
            Dict with parsed components
        """
        query_lower = query.lower().strip()
        
        parsed = {
            'original_query': query,
            'brand': None,
            'product_line': None,
            'series': None,
            'model_number': None,
            'variant': None  # Pro, Ultra, Plus, etc.
        }
        
        # Detect brand
        for brand, config in self.BRAND_CONFIG.items():
            # Check if brand name is in query
            if brand.replace('_', ' ') in query_lower or brand in query_lower:
                parsed['brand'] = brand
                break
            # Check product lines
            for product_line in config.get('product_lines', []):
                if product_line in query_lower:
                    parsed['brand'] = brand
                    parsed['product_line'] = product_line
                    break
            if parsed['brand']:
                break
        
        # Extract product line if not found
        if parsed['brand'] and not parsed['product_line']:
            config = self.BRAND_CONFIG.get(parsed['brand'], {})
            for product_line in config.get('product_lines', []):
                if product_line in query_lower:
                    parsed['product_line'] = product_line
                    break
        
        # Extract series
        if parsed['brand'] and parsed['product_line']:
            config = self.BRAND_CONFIG.get(parsed['brand'], {})
            series_options = config.get('series', {}).get(parsed['product_line'], [])
            for series in series_options:
                if series in query_lower:
                    parsed['series'] = series
                    break
        
        # Extract model number using pattern
        if parsed['brand']:
            config = self.BRAND_CONFIG.get(parsed['brand'], {})
            pattern = config.get('model_pattern', r'\b(\d+)\b')
            matches = re.findall(pattern, query_lower)
            if matches:
                # Get the most significant number (usually the model)
                if isinstance(matches[0], tuple):
                    parsed['model_number'] = ''.join(filter(None, matches[0]))
                else:
                    parsed['model_number'] = matches[0]
        
        # Extract variant (Pro, Ultra, Plus, Max, etc.)
        variants = ['pro max', 'ultra', 'pro', 'plus', 'max', 'lite', 'mini', 'se', 'fe']
        for variant in variants:
            if variant in query_lower:
                parsed['variant'] = variant
                break
        
        return parsed
    
    def parse_product_name(self, product_name: str) -> Dict:
        """Parse scraped product name using same logic"""
        return self.parse_search_query(product_name)
    
    def validate_product(self, search_query: str, product_name: str) -> ProductValidation:
        """
        Validate if a scraped product matches the search query.
        
        Args:
            search_query: Original search query
            product_name: Scraped product name
            
        Returns:
            ProductValidation with match details
        """
        self.validation_stats['total'] += 1
        
        query_parsed = self.parse_search_query(search_query)
        product_parsed = self.parse_product_name(product_name)
        
        # Track matches
        brand_match = True
        series_match = True
        model_match = True
        reasons = []
        
        # 1. Brand validation (lenient - just needs to match)
        if query_parsed['brand'] and product_parsed['brand']:
            if query_parsed['brand'] != product_parsed['brand']:
                brand_match = False
                reasons.append(f"Brand mismatch: expected '{query_parsed['brand']}', got '{product_parsed['brand']}'")
                self.validation_stats['brand_mismatch'] += 1
        
        # 2. Series validation (STRICT - must match exactly)
        if query_parsed['series'] and brand_match:
            if product_parsed['series']:
                if query_parsed['series'].lower() != product_parsed['series'].lower():
                    series_match = False
                    reasons.append(f"Series mismatch: expected '{query_parsed['series']}', got '{product_parsed['series']}'")
                    self.validation_stats['series_mismatch'] += 1
            else:
                # Check if series is in product name
                if query_parsed['series'].lower() not in product_name.lower():
                    series_match = False
                    reasons.append(f"Series '{query_parsed['series']}' not found in product name")
                    self.validation_stats['series_mismatch'] += 1
        
        # 3. Model number validation (STRICT - must match exactly)
        if query_parsed['model_number'] and brand_match:
            if product_parsed['model_number']:
                # Extract numeric parts for comparison
                query_num = re.sub(r'[^0-9]', '', str(query_parsed['model_number']))
                product_num = re.sub(r'[^0-9]', '', str(product_parsed['model_number']))
                
                if query_num and product_num and query_num != product_num:
                    model_match = False
                    reasons.append(f"Model mismatch: expected '{query_parsed['model_number']}', got '{product_parsed['model_number']}'")
                    self.validation_stats['model_mismatch'] += 1
            else:
                # Check if model number is in product name
                if str(query_parsed['model_number']) not in product_name:
                    model_match = False
                    reasons.append(f"Model number '{query_parsed['model_number']}' not found in product name")
                    self.validation_stats['model_mismatch'] += 1
        
        # Calculate confidence
        match_count = sum([brand_match, series_match, model_match])
        confidence = match_count / 3.0
        
        # Determine overall validity
        # Brand must match, and at least one of series/model must match if specified
        is_valid = brand_match and (series_match or model_match)
        
        if not is_valid:
            self.validation_stats['invalid'] += 1
        else:
            self.validation_stats['valid'] += 1
        
        reason = "; ".join(reasons) if reasons else "All checks passed"
        
        return ProductValidation(
            is_valid=is_valid,
            brand_match=brand_match,
            series_match=series_match,
            model_match=model_match,
            confidence=confidence,
            reason=reason
        )
    
    def filter_products(self, search_query: str, products: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter products based on validation.
        
        Args:
            search_query: Original search query
            products: List of scraped product dictionaries
            
        Returns:
            Tuple of (valid_products, invalid_products)
        """
        valid_products = []
        invalid_products = []
        
        for product in products:
            product_name = product.get('name', '')
            validation = self.validate_product(search_query, product_name)
            
            # Add validation info to product
            product['validation'] = {
                'is_valid': validation.is_valid,
                'brand_match': validation.brand_match,
                'series_match': validation.series_match,
                'model_match': validation.model_match,
                'confidence': validation.confidence,
                'reason': validation.reason
            }
            
            if validation.is_valid:
                valid_products.append(product)
            else:
                invalid_products.append(product)
        
        return valid_products, invalid_products
    
    def get_stats(self) -> Dict:
        """Get validation statistics"""
        total = self.validation_stats['total']
        if total > 0:
            return {
                **self.validation_stats,
                'accuracy': self.validation_stats['valid'] / total * 100,
                'rejection_rate': self.validation_stats['invalid'] / total * 100
            }
        return self.validation_stats
    
    def print_stats(self):
        """Print validation statistics"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 PRODUCT VALIDATION STATISTICS")
        print("="*60)
        print(f"Total Products Validated: {stats['total']}")
        print(f"Valid Products: {stats['valid']} ({stats.get('accuracy', 0):.1f}%)")
        print(f"Invalid Products: {stats['invalid']} ({stats.get('rejection_rate', 0):.1f}%)")
        print(f"\nMismatch Breakdown:")
        print(f"  Brand Mismatches: {stats['brand_mismatch']}")
        print(f"  Series Mismatches: {stats['series_mismatch']}")
        print(f"  Model Mismatches: {stats['model_mismatch']}")
        print("="*60)


# Test the validator
if __name__ == "__main__":
    validator = ProductValidator()
    
    # Test cases
    test_cases = [
        # iPhone tests
        ("iPhone 16 Pro Max", "Apple iPhone 16 Pro Max 256GB - Black"),
        ("iPhone 16", "Apple iPhone 17 Pro Max 128GB"),  # Should fail - wrong model
        ("iPhone 16", "Samsung Galaxy S24 Ultra"),  # Should fail - wrong brand
        
        # Samsung tests
        ("Samsung Galaxy S24 Ultra", "Samsung Galaxy S24 Ultra 5G 256GB"),
        ("Samsung Galaxy S24", "Samsung Galaxy S23 FE"),  # Should fail - wrong model
        ("Samsung Galaxy A54", "Samsung Galaxy A55 5G"),  # Should fail - wrong model
        ("Samsung Galaxy M34", "Samsung Galaxy F34 5G"),  # Should fail - wrong series
        
        # Vivo tests
        ("Vivo V30", "Vivo V30 Pro 5G 256GB"),  # Valid - V30 matches
        ("Vivo V30", "Vivo Y100 5G"),  # Should fail - V != Y
        ("Vivo V30", "Vivo T3 Pro 5G"),  # Should fail - V != T
        ("Vivo V30", "Vivo V40 Pro"),  # Should fail - 30 != 40
        
        # OnePlus tests
        ("OnePlus 12", "OnePlus 12 256GB Silky Black"),
        ("OnePlus 12", "OnePlus 11 5G"),  # Should fail
        ("OnePlus Nord CE3", "OnePlus Nord CE 3 Lite"),  # Valid
        
        # Laptop tests
        ("Dell Inspiron 15", "Dell Inspiron 15 3520 Intel Core i5"),
        ("HP Pavilion x360", "HP Pavilion x360 14-inch Touchscreen"),
        ("Lenovo IdeaPad", "Lenovo ThinkPad T14"),  # Should fail - different line
    ]
    
    print("\n" + "🔍"*30)
    print("  PRODUCT VALIDATION TEST CASES")
    print("🔍"*30 + "\n")
    
    for query, product_name in test_cases:
        result = validator.validate_product(query, product_name)
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        print(f"\nQuery: \"{query}\"")
        print(f"Product: \"{product_name[:50]}...\"" if len(product_name) > 50 else f"Product: \"{product_name}\"")
        print(f"Result: {status} (Confidence: {result.confidence:.0%})")
        print(f"Reason: {result.reason}")
        print("-" * 60)
    
    validator.print_stats()
