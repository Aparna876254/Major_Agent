from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import random
import re
import json
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from collections import defaultdict
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import webbrowser
import requests
from io import BytesIO
from threading import Thread
from webdriver_manager.chrome import ChromeDriverManager
import os
from typing import Dict, List, Any, Optional

# Import neural sentiment analyzer
try:
    from neural_sentiment_analyzer import NeuralSentimentAnalyzer, TRANSFORMERS_AVAILABLE
    NEURAL_SENTIMENT_AVAILABLE = TRANSFORMERS_AVAILABLE
except ImportError:
    NEURAL_SENTIMENT_AVAILABLE = False
    print("⚠️ Neural sentiment analyzer not available. Run: pip install transformers torch")

# Import product validator for strict Brand/Series/Model verification
try:
    from product_validator import ProductValidator
    PRODUCT_VALIDATOR_AVAILABLE = True
except ImportError:
    PRODUCT_VALIDATOR_AVAILABLE = False
    print("⚠️ Product validator not available. Strict model verification disabled.")

# Import UMAP analyzer for clustering visualization
try:
    from umap_rag_analyzer import UMAPAnalyzer, RAGStorage, UMAP_AVAILABLE
    UMAP_ANALYZER_AVAILABLE = UMAP_AVAILABLE
except ImportError:
    UMAP_ANALYZER_AVAILABLE = False
    print("⚠️ UMAP analyzer not available. Run: pip install umap-learn")

# Import power monitor for consumption tracking
try:
    from power_monitor import PowerMonitor
    POWER_MONITOR_AVAILABLE = True
except ImportError:
    POWER_MONITOR_AVAILABLE = False
    print("⚠️ Power monitor not available. Run: pip install psutil")

# Import for anti-detection
import string
import pickle
from selenium.webdriver.common.action_chains import ActionChains


# ===========================
# ADVANCED REVIEW SCRAPER
# ===========================

class AdvancedReviewScraper:
    """
    Advanced Review Scraper with Multiple Fallback Strategies
    Handles dynamic loading, pagination, and anti-bot measures
    """
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 15)
    
    def human_like_scroll(self, scrolls=3):
        """Simulate human scrolling behavior"""
        for i in range(scrolls):
            scroll_distance = random.randint(300, 700)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
            time.sleep(random.uniform(0.5, 1.5))
    
    def extract_amazon_reviews_deep(self, product_url, max_reviews=50):
        """
        Extract Amazon reviews with multiple strategies
        Returns: List of review dictionaries
        """
        print(f"🔍 Deep scraping Amazon reviews from: {product_url}")
        
        all_reviews = []
        
        try:
            self.driver.get(product_url)
            time.sleep(random.uniform(2, 4))
            
            # Strategy 1: Click "See All Reviews" button
            try:
                see_all_btn = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 
                        "a[data-hook='see-all-reviews-link-foot'], "
                        "a[href*='customerReviews'], "
                        ".a-link-emphasis[href*='reviews']"))
                )
                see_all_btn.click()
                time.sleep(random.uniform(3, 5))
                print("✅ Navigated to reviews page")
            except:
                print("⚠️ Could not find 'See All Reviews' button, staying on product page")
            
            # Strategy 2: Extract reviews with multiple selectors
            reviews_extracted = 0
            page_num = 1
            
            while reviews_extracted < max_reviews and page_num <= 5:
                print(f"📄 Scraping page {page_num}...")
                
                try:
                    self.wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-hook='review'], .review, .a-section.review")
                    ))
                except TimeoutException:
                    print("⚠️ Reviews not loading, trying alternative selectors")
                
                self.human_like_scroll(scrolls=5)
                time.sleep(2)
                
                review_selectors = [
                    "[data-hook='review']",
                    ".review",
                    ".a-section.review",
                    "[data-hook='review-collapsed']"
                ]
                
                review_elements = []
                for selector in review_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            review_elements = elements
                            print(f"✅ Found {len(elements)} reviews with selector: {selector}")
                            break
                    except:
                        continue
                
                if not review_elements:
                    print("❌ No reviews found on this page")
                    break
                
                for elem in review_elements:
                    if reviews_extracted >= max_reviews:
                        break
                    
                    try:
                        review_data = self._extract_amazon_review_data(elem)
                        if review_data and review_data['text']:
                            all_reviews.append(review_data)
                            reviews_extracted += 1
                    except Exception as e:
                        print(f"⚠️ Failed to extract review: {e}")
                        continue
                
                if reviews_extracted < max_reviews:
                    if not self._click_next_page_amazon():
                        break
                    page_num += 1
                    time.sleep(random.uniform(3, 5))
                else:
                    break
            
            print(f"✅ Successfully extracted {len(all_reviews)} Amazon reviews")
            return all_reviews
            
        except Exception as e:
            print(f"❌ Amazon review scraping failed: {e}")
            return all_reviews
    
    def _extract_amazon_review_data(self, element):
        """Extract structured data from a single review element"""
        review = {
            'text': '',
            'rating': 0,
            'title': '',
            'author': '',
            'date': '',
            'verified': False,
            'helpful_count': 0
        }
        
        try:
            text_selectors = [
                "[data-hook='review-body'] span",
                ".review-text-content span",
                ".a-expander-content.reviewText span"
            ]
            for selector in text_selectors:
                try:
                    text_elem = element.find_element(By.CSS_SELECTOR, selector)
                    review['text'] = text_elem.text.strip()
                    if review['text']:
                        break
                except:
                    continue
            
            try:
                rating_elem = element.find_element(By.CSS_SELECTOR, 
                    "[data-hook='review-star-rating'] span, .review-rating span")
                rating_text = rating_elem.get_attribute('textContent')
                review['rating'] = float(rating_text.split()[0])
            except:
                pass
            
            try:
                title_elem = element.find_element(By.CSS_SELECTOR, 
                    "[data-hook='review-title'], .review-title span")
                review['title'] = title_elem.text.strip()
            except:
                pass
            
            try:
                author_elem = element.find_element(By.CSS_SELECTOR, 
                    "[data-hook='review-author'], .a-profile-name")
                review['author'] = author_elem.text.strip()
            except:
                pass
            
            try:
                date_elem = element.find_element(By.CSS_SELECTOR, 
                    "[data-hook='review-date'], .review-date")
                review['date'] = date_elem.text.strip()
            except:
                pass
            
            try:
                verified = element.find_element(By.CSS_SELECTOR, 
                    "[data-hook='avp-badge'], .a-color-success")
                review['verified'] = 'Verified Purchase' in verified.text
            except:
                pass
            
            try:
                helpful_elem = element.find_element(By.CSS_SELECTOR, 
                    "[data-hook='helpful-vote-statement']")
                helpful_text = helpful_elem.text
                numbers = re.findall(r'\d+', helpful_text)
                if numbers:
                    review['helpful_count'] = int(numbers[0])
            except:
                pass
            
            return review
            
        except Exception as e:
            print(f"⚠️ Error extracting review data: {e}")
            return review
    
    def _click_next_page_amazon(self):
        """Navigate to next review page"""
        try:
            next_selectors = [
                "li.a-last a",
                ".a-pagination .a-last a",
                "a[aria-label='Next page']"
            ]
            
            for selector in next_selectors:
                try:
                    next_btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if 'a-disabled' not in (next_btn.get_attribute('class') or ''):
                        next_btn.click()
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    def extract_flipkart_reviews_deep(self, product_url, max_reviews=50):
        """
        Extract Flipkart reviews with AJAX handling
        Returns: List of review dictionaries
        """
        print(f"🔍 Deep scraping Flipkart reviews from: {product_url}")
        
        all_reviews = []
        
        try:
            self.driver.get(product_url)
            time.sleep(random.uniform(3, 5))
            
            try:
                reviews_section = self.driver.find_element(By.CSS_SELECTOR, 
                    "div[class*='_1YokD2'], div[class*='col._2a50qk'], div.col.JOpGWq")
                self.driver.execute_script("arguments[0].scrollIntoView(true);", reviews_section)
                time.sleep(2)
            except:
                print("⚠️ Could not locate reviews section, scrolling down")
                self.human_like_scroll(scrolls=8)
            
            reviews_extracted = 0
            page_num = 1
            
            while reviews_extracted < max_reviews and page_num <= 5:
                print(f"📄 Scraping page {page_num}...")
                
                try:
                    self.wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div[class*='_1AtVbE'], div.col._2wzgFH")
                    ))
                except:
                    print("⚠️ Reviews not loading")
                
                self.human_like_scroll(scrolls=3)
                time.sleep(2)
                
                review_selectors = [
                    "div[class*='_1AtVbE']",
                    "div.col._2wzgFH",
                    "div[class*='review']"
                ]
                
                review_elements = []
                for selector in review_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            review_elements = elements
                            print(f"✅ Found {len(elements)} reviews with selector: {selector}")
                            break
                    except:
                        continue
                
                if not review_elements:
                    print("❌ No reviews found")
                    break
                
                for elem in review_elements:
                    if reviews_extracted >= max_reviews:
                        break
                    
                    try:
                        review_data = self._extract_flipkart_review_data(elem)
                        if review_data and review_data['text']:
                            all_reviews.append(review_data)
                            reviews_extracted += 1
                    except Exception as e:
                        print(f"⚠️ Failed to extract review: {e}")
                        continue
                
                if reviews_extracted < max_reviews:
                    if not self._click_next_page_flipkart():
                        break
                    page_num += 1
                    time.sleep(random.uniform(3, 5))
                else:
                    break
            
            print(f"✅ Successfully extracted {len(all_reviews)} Flipkart reviews")
            return all_reviews
            
        except Exception as e:
            print(f"❌ Flipkart review scraping failed: {e}")
            return all_reviews
    
    def _extract_flipkart_review_data(self, element):
        """Extract structured data from Flipkart review"""
        review = {
            'text': '',
            'rating': 0,
            'title': '',
            'author': '',
            'date': '',
            'verified': False,
            'helpful_count': 0
        }
        
        try:
            text_selectors = [
                "div[class*='t-ZTKy']",
                "div.t-ZTKy",
                "div[class*='_6K-7Co']"
            ]
            for selector in text_selectors:
                try:
                    text_elem = element.find_element(By.CSS_SELECTOR, selector)
                    review['text'] = text_elem.text.strip()
                    if review['text']:
                        break
                except:
                    continue
            
            try:
                rating_elem = element.find_element(By.CSS_SELECTOR, 
                    "div[class*='_3LWZlK'], div.XQDdHH")
                rating_text = rating_elem.text.strip()
                review['rating'] = float(rating_text.split()[0]) if rating_text else 0
            except:
                pass
            
            try:
                title_elem = element.find_element(By.CSS_SELECTOR, "p[class*='_2-N8zT']")
                review['title'] = title_elem.text.strip()
            except:
                pass
            
            try:
                author_elem = element.find_element(By.CSS_SELECTOR, "p[class*='_2sc7ZR']")
                author_text = author_elem.text.strip()
                review['author'] = author_text
                review['verified'] = 'Certified Buyer' in author_text
            except:
                pass
            
            return review
            
        except Exception as e:
            print(f"⚠️ Error extracting Flipkart review: {e}")
            return review
    
    def _click_next_page_flipkart(self):
        """Navigate to next page on Flipkart"""
        try:
            next_btn = self.driver.find_element(By.CSS_SELECTOR, 
                "a[class*='_1LKTO3']:last-child, nav a:last-child")
            
            if 'disabled' not in (next_btn.get_attribute('class') or ''):
                next_btn.click()
                return True
            return False
        except:
            return False


# ===========================
# CROMA SCRAPER
# ===========================

class CromaScraper:
    """Scraper for Croma.com"""
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 15)
        self.base_url = "https://www.croma.com"
    
    def search_products(self, query, max_products=10):
        """Search for products on Croma"""
        print(f"🔍 Searching Croma for: {query}")
        
        products = []
        
        try:
            search_url = f"{self.base_url}/search/?q={query.replace(' ', '%20')}"
            self.driver.get(search_url)
            time.sleep(random.uniform(3, 5))
            
            try:
                self.wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".product-item, .product, li.plp-card")
                ))
            except TimeoutException:
                print("⚠️ No products found on Croma")
                return products
            
            for _ in range(3):
                self.driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(random.uniform(1, 2))
            
            product_selectors = [
                ".product-item",
                "li.plp-card",
                ".cp-product",
                "div[class*='product']"
            ]
            
            product_elements = []
            for selector in product_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        product_elements = elements[:max_products]
                        print(f"✅ Found {len(product_elements)} products on Croma")
                        break
                except:
                    continue
            
            for elem in product_elements:
                try:
                    product = self._extract_croma_product(elem)
                    if product and product['name'] and product['price']:
                        products.append(product)
                        if len(products) >= max_products:
                            break
                except Exception as e:
                    print(f"⚠️ Failed to extract Croma product: {e}")
                    continue
            
            print(f"✅ Extracted {len(products)} products from Croma")
            return products
            
        except Exception as e:
            print(f"❌ Croma scraping failed: {e}")
            return products
    
    def _extract_croma_product(self, element):
        """Extract product data from Croma listing"""
        product = {
            'name': '',
            'price': 0,
            'price_numeric': 0,
            'original_price': 0,
            'discount': 0,
            'rating': '',
            'reviews': '',
            'image_url': '',
            'product_link': '',
            'source': 'Croma',
            'category': 'Electronics',
            'subcategory': 'General',
            'availability': 'In Stock',
            'emi_available': False,
            'technical_details': {},
            'features': [],
            'description': ''
        }
        
        try:
            name_selectors = ["h3.product-title", ".cp-product__title", "a.product-title", "h3 a"]
            for selector in name_selectors:
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, selector)
                    product['name'] = name_elem.text.strip()
                    if product['name']:
                        break
                except:
                    continue
            
            link_selectors = ["a.product-link", "h3 a", "a[href*='/p/']"]
            for selector in link_selectors:
                try:
                    link_elem = element.find_element(By.CSS_SELECTOR, selector)
                    href = link_elem.get_attribute('href')
                    product['product_link'] = href if href.startswith('http') else self.base_url + href
                    break
                except:
                    continue
            
            price_selectors = [".product-price .amount", ".new-price", "span[class*='price']", ".price-final"]
            for selector in price_selectors:
                try:
                    price_elem = element.find_element(By.CSS_SELECTOR, selector)
                    price_text = price_elem.text.strip()
                    product['price'] = price_text
                    product['price_numeric'] = self._clean_price(price_text)
                    if product['price_numeric'] > 0:
                        break
                except:
                    continue
            
            try:
                original_elem = element.find_element(By.CSS_SELECTOR, ".old-price, .price-old, .product-price .old")
                original_text = original_elem.text.strip()
                product['original_price'] = self._clean_price(original_text)
                
                if product['original_price'] > product['price_numeric']:
                    product['discount'] = round(
                        ((product['original_price'] - product['price_numeric']) / product['original_price']) * 100, 1
                    )
            except:
                product['original_price'] = product['price_numeric']
            
            try:
                rating_elem = element.find_element(By.CSS_SELECTOR, ".product-rating, .rating, [class*='star']")
                rating_text = rating_elem.text.strip()
                numbers = re.findall(r'(\d+\.?\d*)', rating_text)
                if numbers:
                    product['rating'] = numbers[0]
            except:
                pass
            
            img_selectors = ["img.product-image", "img[class*='product']", "img"]
            for selector in img_selectors:
                try:
                    img_elem = element.find_element(By.CSS_SELECTOR, selector)
                    product['image_url'] = img_elem.get_attribute('src') or img_elem.get_attribute('data-src')
                    if product['image_url']:
                        break
                except:
                    continue
            
            try:
                emi_elem = element.find_element(By.CSS_SELECTOR, "[class*='emi'], [class*='EMI']")
                product['emi_available'] = True
            except:
                pass
            
            return product
            
        except Exception as e:
            print(f"⚠️ Error extracting Croma product: {e}")
            return product
    
    def _clean_price(self, price_text):
        """Extract numeric price from text"""
        try:
            cleaned = re.sub(r'[^\d.]', '', price_text)
            return float(cleaned) if cleaned else 0
        except:
            return 0


# ===========================
# RELIANCE DIGITAL SCRAPER
# ===========================

class RelianceDigitalScraper:
    """Scraper for RelianceDigital.in"""
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 15)
        self.base_url = "https://www.reliancedigital.in"
    
    def _generate_search_variations(self, query):
        """Generate multiple search query variations for better results"""
        variations = []
        
        # Original query
        variations.append(query)
        
        # Split camelCase/PascalCase (e.g., OnePlus -> One Plus)
        import re
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', query)
        if spaced != query:
            variations.append(spaced)
        
        # Remove extra spaces and try lowercase
        simple = ' '.join(query.split())
        if simple.lower() not in [v.lower() for v in variations]:
            variations.append(simple)
        
        # Try with spaces between brand words (iPhone15 -> iPhone 15)
        with_spaces = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', query)
        if with_spaces != query and with_spaces not in variations:
            variations.append(with_spaces)
        
        return variations[:3]  # Max 3 variations
    
    def search_products(self, query, max_products=10):
        """Search for products on Reliance Digital"""
        print(f"🔍 Searching Reliance Digital for: {query}")
        
        products = []
        query_variations = self._generate_search_variations(query)
        
        for variation in query_variations:
            try:
                # Use full URL format with search_term and internal_source for better results
                encoded_query = variation.replace(' ', '%20')
                search_url = f"{self.base_url}/products?q={encoded_query}&search_term={encoded_query}&internal_source=search_prompt&page_no=1&page_size=12&page_type=number"
                print(f"📌 Trying Reliance Digital URL: {search_url}")
                self.driver.get(search_url)
                time.sleep(random.uniform(2, 4))
                
                # Dismiss notification popup if present
                try:
                    no_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'No, don')]")
                    no_btn.click()
                    time.sleep(0.5)
                except:
                    pass
                
                try:
                    self.wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, ".product-card, .sp, [class*='product']")
                    ))
                except TimeoutException:
                    print(f"   ⚠️ No products with variation: {variation}")
                    continue
                
                # Scroll to load products
                for _ in range(3):
                    self.driver.execute_script("window.scrollBy(0, 800);")
                    time.sleep(random.uniform(0.5, 1))
                
                # Try to find products
                product_selectors = [".product-card", ".sp", "[data-testid='product-card']"]
                product_elements = []
                
                for selector in product_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            product_elements = elements[:max_products * 3]
                            print(f"   ✅ Found {len(elements)} products with variation: {variation}")
                            break
                    except:
                        continue
                
                if product_elements:
                    break  # Found products, stop trying variations
                    
            except Exception as e:
                print(f"   ⚠️ Error with variation '{variation}': {e}")
                continue
        
        if not product_elements:
            print("⚠️ No products found with any URL variation")
            return products
        
        print(f"✅ Found {len(product_elements)} products on Reliance Digital")
        
        # Build search keywords for relevance check
        query_words = set(query.lower().split())
        # Remove common words
        query_words -= {'the', 'a', 'an', 'for', 'with', 'and', 'or', 'in', 'on', 'at', 'to'}
        
        for elem in product_elements:
            try:
                product = self._extract_reliance_product(elem)
                if product and product['name'] and product['price_numeric']:
                    # Check relevance - at least one keyword should match
                    name_lower = product['name'].lower()
                    is_relevant = any(word in name_lower for word in query_words if len(word) > 2)
                    
                    if is_relevant:
                        products.append(product)
                        if len(products) >= max_products:
                            break
                    else:
                        print(f"   ⚠️ Skipping irrelevant: {product['name'][:40]}...")
            except Exception as e:
                print(f"⚠️ Failed to extract Reliance product: {e}")
                continue
        
        print(f"✅ Extracted {len(products)} relevant products from Reliance Digital")
        return products
    
    def _extract_reliance_product(self, element):
        """Extract product data from Reliance Digital listing"""
        product = {
            'name': '',
            'price': '',
            'price_numeric': 0,
            'original_price': 0,
            'discount': 0,
            'rating': '',
            'reviews': '',
            'image_url': '',
            'product_link': '',
            'source': 'Reliance Digital',
            'category': 'Electronics',
            'subcategory': 'General',
            'availability': 'In Stock',
            'delivery_available': False,
            'technical_details': {},
            'features': [],
            'description': ''
        }
        
        try:
            # Get full text first for filtering
            full_text = element.text.strip()
            
            # Updated name selectors based on current site structure
            name_selectors = ["p[class*='name']", "a p", ".sp__name", ".product-title", "h3", "[class*='title']"]
            for selector in name_selectors:
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, selector)
                    text = name_elem.text.strip()
                    # Skip if it looks like a price or badge
                    if text and not text.startswith('₹') and 'OFF' not in text and 'Compare' not in text:
                        product['name'] = text
                        break
                except:
                    continue
            
            # If no name found, try to extract from full text
            if not product['name'] and full_text:
                lines = [l.strip() for l in full_text.split('\n') if l.strip()]
                for line in lines:
                    # Skip prices, badges, and short text
                    if not line.startswith('₹') and 'OFF' not in line and 'Compare' not in line and len(line) > 10:
                        product['name'] = line
                        break
            
            try:
                link_elem = element.find_element(By.CSS_SELECTOR, "a[href*='/p/'], a[href*='product']")
                href = link_elem.get_attribute('href')
                product['product_link'] = href if href and href.startswith('http') else self.base_url + (href or '')
            except:
                try:
                    link_elem = element.find_element(By.CSS_SELECTOR, "a")
                    href = link_elem.get_attribute('href')
                    product['product_link'] = href if href and href.startswith('http') else self.base_url + (href or '')
                except:
                    pass
            
            # Updated price selectors - look for ₹ symbol
            price_selectors = ["span[class*='price']", "span[class*='Price']", ".sp__price", "[class*='amount']"]
            for selector in price_selectors:
                try:
                    price_elems = element.find_elements(By.CSS_SELECTOR, selector)
                    for price_elem in price_elems:
                        price_text = price_elem.text.strip()
                        if '₹' in price_text or price_text.replace(',', '').replace('.', '').isdigit():
                            product['price'] = price_text
                            product['price_numeric'] = self._clean_price(price_text)
                            if product['price_numeric'] > 0:
                                break
                    if product['price_numeric'] > 0:
                        break
                except:
                    continue
            
            # Fallback: extract price from full text
            if product['price_numeric'] == 0 and full_text:
                import re
                price_match = re.search(r'₹[\d,]+\.?\d*', full_text)
                if price_match:
                    product['price'] = price_match.group()
                    product['price_numeric'] = self._clean_price(price_match.group())
            
            try:
                original_elem = element.find_element(By.CSS_SELECTOR, ".sp__price--strike, .old-price, [class*='strike']")
                original_text = original_elem.text.strip()
                product['original_price'] = self._clean_price(original_text)
                
                if product['original_price'] > product['price_numeric']:
                    product['discount'] = round(
                        ((product['original_price'] - product['price_numeric']) / product['original_price']) * 100, 1
                    )
            except:
                product['original_price'] = product['price_numeric']
            
            try:
                rating_elem = element.find_element(By.CSS_SELECTOR, ".sp__rating, [class*='rating']")
                rating_text = rating_elem.text.strip()
                numbers = re.findall(r'(\d+\.?\d*)', rating_text)
                if numbers:
                    product['rating'] = numbers[0]
            except:
                pass
            
            # Fast image extraction - check multiple attributes for lazy-loaded images
            try:
                img_elem = element.find_element(By.CSS_SELECTOR, "img")
                # Try multiple attributes in order of preference
                for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                    img_url = img_elem.get_attribute(attr)
                    if img_url and img_url.startswith('http') and 'placeholder' not in img_url.lower():
                        product['image_url'] = img_url
                        break
            except:
                pass
            
            return product
            
        except Exception as e:
            print(f"⚠️ Error extracting Reliance product: {e}")
            return product
    
    def _clean_price(self, price_text):
        """Extract numeric price"""
        try:
            cleaned = re.sub(r'[^\d.]', '', price_text)
            return float(cleaned) if cleaned else 0
        except:
            return 0


# ===========================
# STEALTH BROWSER (Anti-Detection)
# ===========================

class StealthBrowser:
    """Anti-detection browser automation - Makes Selenium undetectable"""
    
    def __init__(self, use_proxy=False, proxy_list=None):
        self.options = Options()
        self.use_proxy = use_proxy
        self.proxy_list = proxy_list or []
        self._configure_stealth()
        self.driver = None
        self.session_file = 'browser_session.pkl'
    
    def start(self):
        """Start the stealth browser"""
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        self._apply_stealth_scripts()
        self._load_session()
        return self.driver
    
    def _configure_stealth(self):
        """Configure Chrome options for stealth"""
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        resolutions = ['1920,1080', '1366,768', '1440,900', '1536,864', '1280,720']
        self.options.add_argument(f'--window-size={random.choice(resolutions)}')
        
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--lang=en-IN')
        self.options.add_argument('--disable-webrtc')
        self.options.add_experimental_option('prefs', {
            'intl.accept_languages': 'en-IN,en-US,en',
            'profile.default_content_setting_values.geolocation': 1
        })
        
        if self.use_proxy and self.proxy_list:
            proxy = random.choice(self.proxy_list)
            self.options.add_argument(f'--proxy-server={proxy}')
    
    def _apply_stealth_scripts(self):
        """Apply JavaScript to hide automation"""
        if not self.driver:
            return
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        self.driver.execute_script("window.navigator.chrome = {runtime: {}};")
        self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});")
        self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-IN', 'en-US', 'en']});")
    
    def human_like_mouse_movement(self, element):
        """Simulate human mouse movement to element"""
        if self.driver is None:
            raise RuntimeError("Driver is not initialized. Call start() first.")
        action = ActionChains(self.driver)
        action.move_to_element(element).perform()
        time.sleep(random.uniform(0.1, 0.3))
    
    def human_like_typing(self, element, text):
        """Type like a human with random delays"""
        element.clear()
        for char in text:
            element.send_keys(char)
            if char == ' ':
                time.sleep(random.uniform(0.1, 0.3))
            else:
                time.sleep(random.uniform(0.05, 0.15))
    
    def random_scroll(self, scrolls=3):
        """Scroll like a human"""
        if self.driver is None:
            raise RuntimeError("Driver is not initialized. Call start() first.")
        for _ in range(scrolls):
            distance = random.randint(200, 800)
            chunks = random.randint(3, 7)
            for chunk in range(chunks):
                self.driver.execute_script(f"window.scrollBy(0, {distance // chunks});")
                time.sleep(random.uniform(0.1, 0.3))
            time.sleep(random.uniform(1, 3))
            
            if random.random() < 0.3:
                self.driver.execute_script(f"window.scrollBy(0, -{random.randint(50, 150)});")
                time.sleep(random.uniform(0.5, 1))
    
    def smart_wait(self, min_seconds=2, max_seconds=5):
        """Wait with random delays"""
        base_wait = random.uniform(min_seconds, max_seconds)
        time.sleep(base_wait)
        if random.random() < 0.2:
            time.sleep(random.uniform(2, 5))
    
    def _save_session(self):
        """Save cookies and session data"""
        try:
            if self.driver:
                cookies = self.driver.get_cookies()
                with open(self.session_file, 'wb') as f:
                    pickle.dump(cookies, f)
        except:
            pass
    
    def _load_session(self):
        """Load previous session cookies"""
        try:
            if os.path.exists(self.session_file) and self.driver:
                with open(self.session_file, 'rb') as f:
                    cookies = pickle.load(f)
                
                self.driver.get("https://www.amazon.in")
                for cookie in cookies:
                    try:
                        self.driver.add_cookie(cookie)
                    except:
                        pass
                print("✅ Session restored")
        except:
            pass
    
    def close(self):
        """Save session and close"""
        self._save_session()
        if self.driver:
            self.driver.quit()


# ===========================
# RATE LIMITER
# ===========================

class RateLimiter:
    """Request rate limiting to avoid detection"""
    
    def __init__(self, requests_per_minute=10):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            oldest = min(self.request_times)
            wait_time = 60 - (now - oldest) + random.uniform(1, 3)
            
            if wait_time > 0:
                print(f"⏳ Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        self.request_times.append(time.time())


# ===========================
# SMART RETRY HANDLER
# ===========================

class SmartRetryHandler:
    """Intelligent retry with exponential backoff"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    
    def retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"❌ Failed after {self.max_retries} attempts: {e}")
                    raise
                
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)


# ===========================
# ENHANCED SENTIMENT ANALYZER (Multi-Model)
# ===========================

class EnhancedSentimentAnalyzer:
    """
    Multi-model sentiment analyzer with advanced features:
    - DistilBERT for general sentiment
    - RoBERTa for nuanced analysis (optional)
    - Aspect-based sentiment extraction
    - Key phrase extraction
    """
    
    def __init__(self, use_gpu=True):
        self.device = -1  # CPU by default
        self.models = {}
        self.is_ready = False
        
        if NEURAL_SENTIMENT_AVAILABLE:
            try:
                import torch
                self.device = 0 if (use_gpu and torch.cuda.is_available()) else -1
            except:
                pass
        
        print(f"🚀 Initializing Enhanced Sentiment Analyzer (Device: {'GPU' if self.device == 0 else 'CPU'})")
        
        self.aspects = {
            'quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'solid', 'premium'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient', 'smooth', 'lag'],
            'value': ['worth', 'price', 'expensive', 'cheap', 'value', 'money', 'affordable'],
            'design': ['design', 'look', 'appearance', 'beautiful', 'ugly', 'sleek', 'style'],
            'features': ['feature', 'function', 'capability', 'option', 'setting'],
            'battery': ['battery', 'charge', 'charging', 'power', 'backup'],
            'camera': ['camera', 'photo', 'picture', 'image', 'video'],
            'display': ['screen', 'display', 'resolution', 'brightness', 'color'],
            'service': ['service', 'support', 'warranty', 'delivery', 'shipping']
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load sentiment analysis models"""
        if not NEURAL_SENTIMENT_AVAILABLE:
            print("⚠️ Transformers not available")
            return
        
        try:
            from transformers import pipeline
            
            print("📦 Loading DistilBERT...")
            self.models['distilbert'] = pipeline(
                task="sentiment-analysis",  # type: ignore[arg-type]
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
                truncation=True,
                max_length=512
            )
            
            self.is_ready = True
            print("✅ Sentiment models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.is_ready = False
    
    def analyze_product(self, reviews_list, product_name="", description=""):
        """Comprehensive product sentiment analysis"""
        if not reviews_list and not product_name and not description:
            return self._empty_analysis()
        
        all_texts = []
        
        if product_name:
            all_texts.append(product_name)
        if description:
            all_texts.append(description[:500])
        
        review_texts = []
        for review in reviews_list:
            if isinstance(review, dict):
                text = review.get('text', '')
                title = review.get('title', '')
                combined = f"{title} {text}".strip()
                if combined:
                    review_texts.append(combined)
                    all_texts.append(combined)
        
        if not all_texts:
            return self._empty_analysis()
        
        results = {
            'overall_sentiment': '',
            'confidence': 0.0,
            'sentiment_distribution': {},
            'aspect_sentiments': {},
            'key_phrases': {'positive': [], 'negative': []},
            'review_breakdown': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': len(review_texts)},
            'recommendation_score': 0.0,
            'summary': ''
        }
        
        # Analyze overall sentiment
        ensemble_scores = self._ensemble_sentiment(all_texts[:20])
        results['overall_sentiment'] = ensemble_scores['label']
        results['confidence'] = ensemble_scores['score']
        results['sentiment_distribution'] = ensemble_scores['distribution']
        
        # Aspect-based sentiment
        if review_texts:
            results['aspect_sentiments'] = self._aspect_sentiment_analysis(review_texts)
        
        # Key phrases
        results['key_phrases'] = self._extract_key_phrases(review_texts)
        
        # Review breakdown
        results['review_breakdown'] = self._categorize_reviews(reviews_list)
        
        # Recommendation score
        results['recommendation_score'] = self._calculate_recommendation_score(results)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _ensemble_sentiment(self, texts):
        """Combine predictions from models"""
        scores = {'positive': [], 'neutral': [], 'negative': []}
        
        for text in texts:
            if not text.strip():
                continue
            
            if 'distilbert' in self.models:
                try:
                    result = self.models['distilbert'](text[:512])[0]
                    label = result['label'].lower()
                    score = result['score']
                    
                    if label == 'positive':
                        scores['positive'].append(score)
                        scores['negative'].append(0)
                    else:
                        scores['negative'].append(score)
                        scores['positive'].append(0)
                except:
                    pass
        
        import numpy as np
        avg_scores = {
            'positive': np.mean(scores['positive']) if scores['positive'] else 0,
            'negative': np.mean(scores['negative']) if scores['negative'] else 0,
            'neutral': np.mean(scores['neutral']) if scores['neutral'] else 0
        }
        
        max_sentiment = max(avg_scores, key=lambda k: avg_scores[k])
        
        return {
            'label': max_sentiment,
            'score': avg_scores[max_sentiment],
            'distribution': avg_scores
        }
    
    def _aspect_sentiment_analysis(self, review_texts):
        """Analyze sentiment for specific product aspects"""
        aspect_sentiments = {}
        
        for aspect, keywords in self.aspects.items():
            relevant_sentences = []
            
            for review in review_texts:
                sentences = re.split(r'[.!?]', review)
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in keywords):
                        relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                aspect_analysis = self._ensemble_sentiment(relevant_sentences[:10])
                aspect_sentiments[aspect] = {
                    'sentiment': aspect_analysis['label'],
                    'confidence': aspect_analysis['score'],
                    'mention_count': len(relevant_sentences)
                }
        
        return aspect_sentiments
    
    def _extract_key_phrases(self, review_texts):
        """Extract positive and negative key phrases"""
        positive_phrases = []
        negative_phrases = []
        
        positive_words = ['excellent', 'great', 'amazing', 'perfect', 'love', 'best', 
                         'awesome', 'fantastic', 'superb', 'good', 'nice', 'happy']
        negative_words = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 
                         'disappointed', 'useless', 'waste', 'defective', 'broken']
        
        for review in review_texts[:20]:
            sentences = re.split(r'[.!?]', review.lower())
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                if any(word in sentence for word in positive_words):
                    if len(sentence) <= 100:
                        positive_phrases.append(sentence.capitalize())
                elif any(word in sentence for word in negative_words):
                    if len(sentence) <= 100:
                        negative_phrases.append(sentence.capitalize())
        
        return {
            'positive': list(set(positive_phrases))[:5],
            'negative': list(set(negative_phrases))[:5]
        }
    
    def _categorize_reviews(self, reviews_list):
        """Categorize reviews as positive/neutral/negative"""
        breakdown = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': len(reviews_list)}
        
        for review in reviews_list:
            if isinstance(review, dict):
                rating = review.get('rating', 0)
                if isinstance(rating, str):
                    try:
                        rating = float(rating.split()[0])
                    except:
                        rating = 0
                
                if rating >= 4:
                    breakdown['positive'] += 1
                elif rating >= 2.5:
                    breakdown['neutral'] += 1
                else:
                    breakdown['negative'] += 1
        
        return breakdown
    
    def _calculate_recommendation_score(self, results):
        """Calculate overall recommendation score (0-100)"""
        score = 50
        
        if results['overall_sentiment'] == 'positive':
            score += 30 * results['confidence']
        elif results['overall_sentiment'] == 'negative':
            score -= 30 * results['confidence']
        
        breakdown = results['review_breakdown']
        if breakdown['total'] > 0:
            positive_ratio = breakdown['positive'] / breakdown['total']
            negative_ratio = breakdown['negative'] / breakdown['total']
            score += 20 * positive_ratio - 20 * negative_ratio
        
        return max(0, min(100, round(score, 1)))
    
    def _generate_summary(self, results):
        """Generate human-readable summary"""
        sentiment = results['overall_sentiment'].capitalize()
        confidence = results['confidence']
        rec_score = results['recommendation_score']
        
        breakdown = results['review_breakdown']
        total = breakdown['total']
        
        summary = f"Overall Sentiment: {sentiment} ({confidence:.1%} confidence). "
        
        if total > 0:
            summary += f"Based on {total} reviews: {breakdown['positive']} positive, "
            summary += f"{breakdown['neutral']} neutral, {breakdown['negative']} negative. "
        
        if rec_score >= 70:
            summary += "✅ Highly Recommended"
        elif rec_score >= 50:
            summary += "👍 Recommended with considerations"
        else:
            summary += "⚠️ Consider alternatives"
        
        return summary
    
    def _empty_analysis(self):
        """Return empty analysis structure"""
        return {
            'overall_sentiment': 'unknown',
            'confidence': 0.0,
            'sentiment_distribution': {},
            'aspect_sentiments': {},
            'key_phrases': {'positive': [], 'negative': []},
            'review_breakdown': {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0},
            'recommendation_score': 50.0,
            'summary': 'Insufficient data for analysis'
        }


# ===========================
# RAG Storage System (Enhanced)
# ===========================

class ProductRAGStorage:
    """RAG-based storage system for product data with semantic search"""
    
    def __init__(self, storage_file='product_rag_db.pkl'):
        self.storage_file = storage_file
        self.products = []
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.vectors = None
        self.load_storage()
    
    def add_product(self, product_data):
        """Add product to RAG storage"""
        product_data['timestamp'] = datetime.now().isoformat()
        product_data['id'] = f"{product_data['source']}_{len(self.products)}"
        self.products.append(product_data)
        self._update_vectors()
        self.save_storage()
    
    def add_products_batch(self, products_list):
        """Add multiple products at once"""
        for product in products_list:
            product['timestamp'] = datetime.now().isoformat()
            product['id'] = f"{product.get('source', 'unknown')}_{len(self.products)}"
            self.products.append(product)
        self._update_vectors()
        self.save_storage()
    
    def _update_vectors(self):
        """Update TF-IDF vectors for semantic search"""
        if not self.products:
            return
        
        texts = []
        for p in self.products:
            # Include all text fields for better semantic search
            text_parts = [
                p.get('name', ''),
                p.get('subcategory', ''),
                p.get('category', ''),
                str(p.get('technical_details', {})),
                str(p.get('additional_info', {})),
                p.get('description', '')
            ]
            texts.append(' '.join(text_parts))
        
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def semantic_search(self, query, top_k=10):
        """Search products using semantic similarity"""
        if not self.products or self.vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            product = self.products[idx].copy()
            product['similarity_score'] = float(similarities[idx])
            results.append(product)
        
        return results
    
    def get_statistics(self):
        """Get statistics about stored products"""
        if not self.products:
            return {}
        
        stats = {
            'total_products': len(self.products),
            'by_source': defaultdict(int),
            'by_category': defaultdict(int),
            'price_stats': {},
            'rating_stats': {},
            'detailed_products': 0
        }
        
        prices = []
        ratings = []
        
        for p in self.products:
            stats['by_source'][p.get('source', 'unknown')] += 1
            stats['by_category'][p.get('category', 'unknown')] += 1
            
            if p.get('technical_details'):
                stats['detailed_products'] += 1
            
            price = p.get('price_numeric', 0)
            if price > 0:
                prices.append(price)
            
            rating = self._extract_rating(p.get('rating', ''))
            if rating > 0:
                ratings.append(rating)
        
        if prices:
            stats['price_stats'] = {
                'min': min(prices),
                'max': max(prices),
                'avg': np.mean(prices),
                'median': np.median(prices)
            }
        
        if ratings:
            stats['rating_stats'] = {
                'min': min(ratings),
                'max': max(ratings),
                'avg': np.mean(ratings),
                'median': np.median(ratings)
            }
        
        return stats
    
    def _extract_rating(self, rating_str):
        """Extract numeric rating from string"""
        if not rating_str:
            return 0.0
        match = re.search(r'(\d+\.?\d*)', str(rating_str))
        return float(match.group(1)) if match else 0.0
    
    def save_storage(self):
        """Save storage to disk"""
        data = {
            'products': self.products,
            'vectorizer': self.vectorizer,
            'vectors': self.vectors
        }
        with open(self.storage_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load_storage(self):
        """Load storage from disk"""
        try:
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
                self.products = data.get('products', [])
                self.vectorizer = data.get('vectorizer', TfidfVectorizer(max_features=500))
                self.vectors = data.get('vectors')
            print(f"Loaded {len(self.products)} products from storage")
        except FileNotFoundError:
            print("No existing storage found, starting fresh")
        except Exception as e:
            print(f"Error loading storage: {e}")
    
    def export_to_csv(self, filename='products_export.csv'):
        """Export products to CSV"""
        if not self.products:
            print("No products to export")
            return
        
        # Flatten nested dictionaries for CSV
        flattened = []
        for p in self.products:
            flat = p.copy()
            
            # Flatten technical_details
            if 'technical_details' in flat and isinstance(flat['technical_details'], dict):
                for k, v in flat['technical_details'].items():
                    clean_key = k.replace(' - ', '_').replace(' ', '_').replace('/', '_')
                    flat[f'spec_{clean_key}'] = v
                del flat['technical_details']
            
            # Flatten additional_info
            if 'additional_info' in flat and isinstance(flat['additional_info'], dict):
                for k, v in flat['additional_info'].items():
                    clean_key = k.replace(' - ', '_').replace(' ', '_').replace('/', '_')
                    flat[f'info_{clean_key}'] = v
                del flat['additional_info']
            
            # Convert features list to string
            if 'features' in flat and isinstance(flat['features'], list):
                flat['features'] = ' | '.join(flat['features'])
            
            flattened.append(flat)
        
        df = pd.DataFrame(flattened)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Exported {len(self.products)} products to {filename}")
    
    
    def clear_storage(self):
        """Clear all stored products"""
        self.products = []
        self.vectors = None
        self.save_storage()

    def delete_product(self, product_id):
        """Delete a product by its ID."""
        initial_count = len(self.products)
        self.products = [p for p in self.products if p.get('id') != product_id]
        if len(self.products) < initial_count:
            self._update_vectors()
            self.save_storage()
            return True
        return False

    def get_product_by_id(self, product_id):
        """Get a single product by its ID."""
        for p in self.products:
            if p.get('id') == product_id:
                return p
        return None

# ===========================
# Enhanced Web Scraping
# ===========================

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

class Browser:
    """A wrapper for the Selenium browser driver."""
    def __init__(self):
        self._driver = None

    def start(self):
        """Starts the browser driver."""
        options = Options()
        options.add_argument(f'user-agent={random.choice(user_agents)}')
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option(
            "prefs", {"profile.default_content_setting_values.notifications": 2}
        )
        
        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=options)
        
        # Avoid basic bot detection
        try:
            self._driver.execute_cdp_cmd(
                "Network.setUserAgentOverride",
                {"userAgent": random.choice(user_agents)},
            )
        except Exception:
            pass
        
        try:
            self._driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
        except Exception:
            pass
        
        self._driver.set_page_load_timeout(60)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying driver."""
        if self._driver is not None:
            return getattr(self._driver, name)
        raise AttributeError(f"'Browser' object has no attribute '{name}' before 'start()' is called.")

    def quit(self):
        """Quits the browser driver."""
        if self._driver:
            self._driver.quit()

def get_browser():
    """Returns a Browser instance."""
    return Browser()


def setup_driver():
    """Create browser via adapter (Selenium-only)."""
    browser = get_browser()
    browser.start()
    return browser


def build_amazon_search_url(product_name):
    """Build Amazon search URL from product name."""
    return f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"


def handle_amazon_interstitial(driver):
    """Handle Amazon interstitial/503 pages. Returns True if blocked."""
    try:
        page_text = driver.page_source.lower()
        if '503' in page_text or 'service unavailable' in page_text or 'robot' in page_text:
            print("Amazon: Detected interstitial/503 page")
            time.sleep(3)
            return True
        return False
    except Exception:
        return False


def handle_flipkart_interstitial(driver):
    """Handle Flipkart interstitial/popup pages. Returns True if blocked."""
    try:
        # Close login popup if present
        try:
            close_btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '✕')]"))
            )
            close_btn.click()
            time.sleep(1)
        except:
            pass
        
        page_text = driver.page_source.lower()
        if '503' in page_text or 'service unavailable' in page_text:
            print("Flipkart: Detected interstitial/503 page")
            time.sleep(3)
            return True
        return False
    except Exception:
        return False


def scrape_amazon_product_details(driver, product_url):
    """Scrape detailed information from Amazon product page"""
    try:
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': [],
            'customer_reviews': [],
            'review_summary': '',
            'rating_breakdown': {}
        }
        
        print(f"      Searching for Amazon product details...")
        
        # ========== CUSTOMER REVIEWS EXTRACTION ==========
        print(f"      Extracting customer reviews...")
        
        # Get rating breakdown (5 star: 61%, 4 star: 22%, etc.)
        try:
            rating_table = driver.find_element(By.ID, "histogramTable")
            rating_rows = rating_table.find_elements(By.CSS_SELECTOR, "tr.a-histogram-row")
            for row in rating_rows:
                try:
                    star_text = row.find_element(By.CSS_SELECTOR, "td.aok-nowrap span").text.strip()
                    percent_text = row.find_element(By.CSS_SELECTOR, "td.a-text-right span").text.strip()
                    if star_text and percent_text:
                        details['rating_breakdown'][star_text] = percent_text
                except:
                    continue
            print(f"      Found rating breakdown: {details['rating_breakdown']}")
        except:
            # Alternative method
            try:
                histogram = driver.find_elements(By.CSS_SELECTOR, "#averageCustomerReviews .a-histogram-row, [data-hook='rating-histogram'] tr")
                for row in histogram:
                    text = row.text.strip()
                    if 'star' in text.lower() and '%' in text:
                        parts = text.split()
                        for i, p in enumerate(parts):
                            if 'star' in p.lower() and i > 0:
                                star = parts[i-1] + ' star'
                                for pct in parts:
                                    if '%' in pct:
                                        details['rating_breakdown'][star] = pct
                                        break
            except:
                pass
        
        # Get "Customers say" summary
        try:
            customers_say = driver.find_element(By.CSS_SELECTOR, "[data-hook='cr-summarization-attributes-list'], #cr-summarization-attributes-list")
            details['review_summary'] = customers_say.text.strip()
            print(f"      Found 'Customers say' summary")
        except:
            try:
                insights = driver.find_element(By.CSS_SELECTOR, ".cr-widget-FocalReviews, [data-hook='lighthut-terms-list']")
                details['review_summary'] = insights.text.strip()
            except:
                pass
        
        # Get top customer reviews
        try:
            review_elements = driver.find_elements(By.CSS_SELECTOR, "[data-hook='review'], .review, .a-section.review")[:5]
            for review in review_elements:
                try:
                    review_data = {}
                    try:
                        title = review.find_element(By.CSS_SELECTOR, "[data-hook='review-title'], .review-title").text.strip()
                        review_data['title'] = title
                    except:
                        pass
                    try:
                        body = review.find_element(By.CSS_SELECTOR, "[data-hook='review-body'], .review-text, .review-text-content").text.strip()
                        review_data['text'] = body
                    except:
                        pass
                    try:
                        rating = review.find_element(By.CSS_SELECTOR, "[data-hook='review-star-rating'], .review-rating").text.strip()
                        review_data['rating'] = rating
                    except:
                        pass
                    if review_data.get('text'):
                        details['customer_reviews'].append(review_data)
                except:
                    continue
            print(f"      Found {len(details['customer_reviews'])} customer reviews")
        except:
            pass
        
        # ========== ORIGINAL DETAIL EXTRACTION ==========
        # Search for sections containing keywords: "detail", "product information", "specification"
        detail_keywords = ['detail', 'product information', 'specification', 'technical', 'product details']
        
        # Method 1: Try specific IDs first
        # Technical Details by ID
        try:
            tech_table = driver.find_element(By.ID, "productDetails_techSpec_section_1")
            rows = tech_table.find_elements(By.TAG_NAME, "tr")
            print(f"      Found Technical Details table with {len(rows)} rows")
            for row in rows:
                try:
                    th = row.find_element(By.TAG_NAME, "th").text.strip()
                    td = row.find_element(By.TAG_NAME, "td").text.strip()
                    if th and td:
                        details['technical_details'][th] = td
                except:
                    continue
        except:
            print(f"      Technical Details table not found by ID, trying alternative methods...")
        
        # Method 2: Search by text for sections containing "detail" or "information"
        if not details['technical_details']:
            try:
                # Find all tables and sections
                all_tables = driver.find_elements(By.TAG_NAME, "table")
                print(f"      Found {len(all_tables)} tables on page, searching for details...")
                
                for table in all_tables:
                    try:
                        # Check if table or its parent contains detail keywords
                        table_text = table.text.lower()
                        if any(keyword in table_text for keyword in detail_keywords):
                            rows = table.find_elements(By.TAG_NAME, "tr")
                            for row in rows:
                                try:
                                    cells = row.find_elements(By.TAG_NAME, "th")
                                    cells.extend(row.find_elements(By.TAG_NAME, "td"))
                                    if len(cells) >= 2:
                                        key = cells[0].text.strip()
                                        value = cells[1].text.strip()
                                        if key and value and len(key) < 100:
                                            details['technical_details'][key] = value
                                except:
                                    continue
                    except:
                        continue
            except:
                pass
        
        # Additional Information
        try:
            additional_table = driver.find_element(By.ID, "productDetails_detailBullets_sections1")
            rows = additional_table.find_elements(By.TAG_NAME, "tr")
            print(f"      Found Additional Information with {len(rows)} rows")
            for row in rows:
                try:
                    th = row.find_element(By.TAG_NAME, "th").text.strip()
                    td = row.find_element(By.TAG_NAME, "td").text.strip()
                    if th and td:
                        details['additional_info'][th] = td
                except:
                    continue
        except:
            pass
        
        # Product Description - search for "description" keyword
        desc_found = False
        try:
            desc_element = driver.find_element(By.ID, "feature-bullets")
            features = desc_element.find_elements(By.TAG_NAME, "li")
            print(f"      Found {len(features)} product features")
            for feat in features:
                text = feat.text.strip()
                if text:
                    details['features'].append(text)
            details['description'] = ' | '.join(details['features'])
            desc_found = True
        except:
            pass
        
        # Alternative description locations - search by keyword
        if not desc_found:
            try:
                desc = driver.find_element(By.ID, "productDescription")
                details['description'] = desc.text.strip()
                print(f"      Found product description")
                desc_found = True
            except:
                pass
        
        # Method 3: Search all divs for "description" or "detail" text
        if not desc_found:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    try:
                        # Check class and id for description keywords
                        div_class = div.get_attribute("class") or ""
                        div_id = div.get_attribute("id") or ""
                        
                        if any(kw in div_class.lower() or kw in div_id.lower() 
                               for kw in ['description', 'detail', 'product-info']):
                            text = div.text.strip()
                            if text and len(text) > 50 and len(text) < 2000:
                                details['description'] = text
                                print(f"      Found description in div: {div_id or div_class[:30]}")
                                break
                    except:
                        continue
            except:
                pass
        
        print(f"      Amazon extraction complete: {len(details['technical_details'])} specs, "
              f"{len(details['additional_info'])} additional info, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Amazon details: {e}")
        import traceback
        traceback.print_exc()
        return None


def scrape_flipkart_product_details(driver, product_url):
    """Scrape detailed information from Flipkart product page - using actual Flipkart selectors"""
    try:
        if product_url:
            driver.get(product_url)
            print(f"      Waiting for page to fully load...")
            time.sleep(5)  # Initial wait for page load
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': [],
            'page_price': '',
            'customer_reviews': [],
            'review_summary': '',
            'rating_breakdown': {},
            'rating': '',
            'category_ratings': {}  # Camera, Battery, Display, Design ratings
        }
        
        print(f"      Searching for Flipkart product details...")
        
        # Scroll slowly to load all sections including reviews
        print(f"      Scrolling to load reviews section...")
        for i in range(8):
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(0.8)  # Wait for lazy-loaded content
        
        # Scroll back up a bit to the reviews section
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
        time.sleep(2)  # Wait for reviews to render
        
        # Try to scroll to Ratings & Reviews section specifically
        try:
            # Look for "Ratings & Reviews" heading and scroll to it
            review_heading = driver.find_element(By.XPATH, "//*[contains(text(), 'Ratings & Reviews') or contains(text(), 'Rating')]")
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});")
            time.sleep(2)  # Wait after scrolling to reviews
        except:
            pass
        
        # ========== RATING EXTRACTION ==========
        # From screenshot: span.PvbNMB contains "2,70,519 Ratings & 9,466 Reviews"
        try:
            rating_selectors = [
                "span.PvbNMB",
                "div._3LWZlK",
                "span._1lRcqv",
                "[class*='rating']"
            ]
            for sel in rating_selectors:
                try:
                    rating_elem = driver.find_element(By.CSS_SELECTOR, sel)
                    rating_text = rating_elem.text.strip()
                    if rating_text:
                        import re
                        match = re.search(r'(\d+\.?\d*)\s*★?', rating_text)
                        if match:
                            details['rating'] = match.group(1)
                            print(f"      Found rating: {details['rating']}")
                            break
                except:
                    continue
        except:
            pass
        
        # ========== RATINGS & REVIEWS SECTION ==========
        # From screenshot: div.KAdfFz "Ratings & Reviews" section
        print(f"      Extracting Flipkart customer reviews...")
        
        # Get rating breakdown (5★, 4★, 3★, 2★, 1★ counts)
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            for star in ['5', '4', '3', '2', '1']:
                patterns = [
                    rf'{star}\s*★\s*([\d,]+)',
                    rf'{star}\s*star[s]?\s*([\d,]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        details['rating_breakdown'][f"{star} Star"] = match.group(1)
                        break
            
            if details['rating_breakdown']:
                print(f"      Found rating breakdown: {details['rating_breakdown']}")
        except:
            pass
        
        # Get category ratings (Camera, Battery, Display, Design)
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            categories = ['Camera', 'Battery', 'Display', 'Design', 'Performance', 'Value']
            for cat in categories:
                pattern = rf'(\d+\.?\d*)\s*\n?\s*{cat}'
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    details['category_ratings'][cat] = match.group(1)
            
            if details['category_ratings']:
                print(f"      Found category ratings: {details['category_ratings']}")
        except:
            pass
        
        # Get customer reviews - extract actual comment text for sentiment analysis
        # From screenshot: div.a6dZNm.mIW33x contains the actual review text like "So beautiful, so elegant..."
        print(f"      Waiting for reviews to load...")
        time.sleep(3)  # Extra wait for reviews section
        
        try:
            # METHOD 1: Direct search for review text elements (most reliable)
            # From screenshot: The actual review text is in div.a6dZNm.mIW33x
            review_text_selectors = [
                "div.a6dZNm.mIW33x",      # Exact class from screenshot
                "div.a6dZNm",              # Parent class
                "div.ZmyHeo div",          # Review content area
                "div.t-ZTKy",              # Alternative review text
                "div._6K-7Co"              # Another review text class
            ]
            
            for text_sel in review_text_selectors:
                try:
                    text_elements = driver.find_elements(By.CSS_SELECTOR, text_sel)
                    print(f"      Trying selector '{text_sel}': found {len(text_elements)} elements")
                    
                    for elem in text_elements[:15]:
                        try:
                            text = elem.text.strip()
                            # Filter: actual review text is usually 20+ chars, not UI elements
                            if text and len(text) > 20 and len(text) < 2000:
                                # Skip navigation/UI text
                                skip_words = ['add to cart', 'buy now', 'login', 'sign in', 'see all', 
                                             'read more', 'helpful', 'report', 'ratings', 'reviews']
                                if not any(skip in text.lower()[:50] for skip in skip_words):
                                    details['customer_reviews'].append({'text': text})
                                    print(f"      ✓ Found review: {text[:60]}...")
                        except:
                            continue
                    
                    if len(details['customer_reviews']) >= 5:
                        break
                except Exception as e:
                    continue
            
            # METHOD 2: Find review containers and extract text from them
            if not details['customer_reviews']:
                print(f"      Trying container-based extraction...")
                container_selectors = [
                    "div.col.EPCmJX",
                    "div._27M-vq", 
                    "div.row._3fWwat"
                ]
                
                for container_sel in container_selectors:
                    containers = driver.find_elements(By.CSS_SELECTOR, container_sel)[:10]
                    for container in containers:
                        try:
                            # Get all text from the container
                            full_text = container.text.strip()
                            lines = full_text.split('\n')
                            
                            # The review text is usually the longest line that's not a name/date
                            for line in lines:
                                line = line.strip()
                                if len(line) > 30 and len(line) < 1000:
                                    # Skip obvious non-review content
                                    if not any(skip in line.lower() for skip in ['certified buyer', 'months ago', 'helpful', '★']):
                                        details['customer_reviews'].append({'text': line})
                                        break
                        except:
                            continue
                    
                    if details['customer_reviews']:
                        break
            
            # METHOD 3: Regex extraction from page source as fallback
            if not details['customer_reviews']:
                print(f"      Trying page text extraction...")
                try:
                    page_text = driver.find_element(By.TAG_NAME, "body").text
                    
                    # Look for common review patterns (sentences after star ratings)
                    import re
                    # Pattern: text between review indicators
                    review_patterns = [
                        r'Fabulous[!\s]+(.{20,200})',
                        r'Awesome[!\s]+(.{20,200})',
                        r'Amazing[!\s]+(.{20,200})',
                        r'Great[!\s]+(.{20,200})',
                        r'Good[!\s]+(.{20,200})',
                        r'Nice[!\s]+(.{20,200})',
                        r'Perfect[!\s]+(.{20,200})',
                        r'Excellent[!\s]+(.{20,200})',
                        r'Best[!\s]+(.{20,200})',
                        r'Worth[!\s]+(.{20,200})',
                    ]
                    
                    for pattern in review_patterns:
                        matches = re.findall(pattern, page_text, re.IGNORECASE)
                        for match in matches[:2]:
                            if len(match) > 20:
                                # Clean up the match
                                clean_text = match.split('\n')[0].strip()
                                if clean_text and len(clean_text) > 20:
                                    details['customer_reviews'].append({'text': clean_text})
                        
                        if len(details['customer_reviews']) >= 5:
                            break
                except:
                    pass
            
            print(f"      Found {len(details['customer_reviews'])} customer reviews for sentiment analysis")
            
            # Print sample reviews for verification
            for i, review in enumerate(details['customer_reviews'][:3]):
                print(f"      Review {i+1}: {review.get('text', '')[:80]}...")
                
        except Exception as e:
            print(f"      Error extracting reviews: {e}")
        
        # ========== SPECIFICATIONS EXTRACTION ==========
        # From screenshot: div.xdON2G "Specifications", div.d2eoIN "Other Details"
        try:
            spec_section_selectors = [
                "div.xdON2G",
                "div.GNDEQ-",
                "div._14cfVK",
                "div[class*='specification']"
            ]
            
            for container_sel in spec_section_selectors:
                try:
                    containers = driver.find_elements(By.CSS_SELECTOR, container_sel)
                    for container in containers:
                        section_title = "Specifications"
                        try:
                            title_elem = container.find_element(By.CSS_SELECTOR, 
                                "div.d2eoIN, div._4BJ2V\\+, [class*='title']")
                            section_title = title_elem.text.strip()[:50] or "Specifications"
                        except:
                            pass
                        
                        rows = container.find_elements(By.CSS_SELECTOR, "tr")
                        for row in rows:
                            try:
                                cells = row.find_elements(By.CSS_SELECTOR, "td")
                                if len(cells) >= 2:
                                    key = cells[0].text.strip()
                                    value = cells[1].text.strip()
                                    if key and value and len(key) < 80:
                                        details['technical_details'][key] = value
                            except:
                                continue
                    
                    if len(details['technical_details']) >= 5:
                        break
                except:
                    continue
            
            print(f"      Found {len(details['technical_details'])} specifications")
        except:
            pass
        
        # ========== HIGHLIGHTS/FEATURES EXTRACTION ==========
        try:
            highlight_selectors = [
                "div._1mXcCf li",
                "div.qFfOgN li",
                "li._21Ahn-",
                "ul._1D2qrc li",
                "[class*='highlight'] li"
            ]
            
            for sel in highlight_selectors:
                try:
                    highlights = driver.find_elements(By.CSS_SELECTOR, sel)
                    for h in highlights[:15]:
                        text = h.text.strip()
                        if text and len(text) > 5 and text not in details['features']:
                            details['features'].append(text)
                    if details['features']:
                        print(f"      Found {len(details['features'])} highlights/features")
                        break
                except:
                    continue
        except:
            pass
        
        # ========== PRODUCT DESCRIPTION EXTRACTION ==========
        try:
            desc_selectors = [
                "div.KgDEGp",
                "div.RmoJUa",
                "div._1mXcCf",
                "div[class*='description']"
            ]
            
            for sel in desc_selectors:
                try:
                    desc_elem = driver.find_element(By.CSS_SELECTOR, sel)
                    text = desc_elem.text.strip()
                    if text and len(text) > 50:
                        if 'Product Description' in text:
                            text = text.replace('Product Description', '').strip()
                        details['description'] = text[:2000]
                        print(f"      Found product description ({len(details['description'])} chars)")
                        break
                except:
                    continue
        except:
            pass
        
        # Combine features into description if empty
        if not details['description'] and details['features']:
            details['description'] = ' | '.join(details['features'])
        
        # ========== PRICE EXTRACTION ==========
        try:
            price_selectors = [
                "div.Nx9bqj._4b5DiR",
                "div._30jeq3._16Jk6d",
                "div.Nx9bqj",
                "div._30jeq3"
            ]
            
            for sel in price_selectors:
                try:
                    price_elem = driver.find_element(By.CSS_SELECTOR, sel)
                    price_text = price_elem.text.strip()
                    if price_text and '₹' in price_text:
                        details['page_price'] = price_text
                        print(f"      Found price: {price_text}")
                        break
                except:
                    continue
        except:
            pass
        
        print(f"      Flipkart extraction complete: {len(details['technical_details'])} specs, {len(details['features'])} features, {len(details['customer_reviews'])} reviews")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Flipkart details: {e}")
        import traceback
        traceback.print_exc()
        return None


def scrape_detailed_amazon(driver, product_name, max_products=5):
    """Enhanced Amazon scraper with deep product details"""
    try:
        driver.get("https://www.amazon.in")
        time.sleep(random.uniform(3, 5))
        
        url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
        driver.get(url)
        time.sleep(random.uniform(4, 6))
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']"))
        )
        time.sleep(2)
    except TimeoutException:
        print("Amazon: Could not load results")
        return []
    except Exception as e:
        print(f"Amazon error: {str(e)}")
        return []
    
    products = []
    items = driver.find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
    
    print(f"Amazon: Found {len(items)} items, scraping details for top {max_products}...")
    
    for idx, item in enumerate(items[:max_products]):
        try:
            print(f"  Scraping Amazon product {idx+1}/{max_products}...")
            
            # Get product link with multiple selector attempts
            product_link = ""
            link_selectors = [
                "h2 a",
                "h2.a-size-mini a",
                ".a-link-normal.s-no-outline",
                ".a-link-normal.a-text-normal",
                "a.a-link-normal[href*='/dp/']",
                "a[href*='/dp/']"
            ]
            
            for selector in link_selectors:
                try:
                    link_element = item.find_element(By.CSS_SELECTOR, selector)
                    product_link = link_element.get_attribute("href")
                    if product_link and '/dp/' in product_link:
                        print(f"    Found link with selector '{selector}': {product_link[:60]}...")
                        break
                except NoSuchElementException:
                    continue
            
            if not product_link:
                print(f"    Failed to extract link: No valid link found with any selector")
                continue
            
            # Get product name with multiple selector attempts
            name = ""
            name_selectors = [
                "h2 span",
                "h2 a span",
                "h2.a-size-mini span.a-size-medium",
                "h2.a-size-mini span.a-size-base-plus",
                "span.a-size-medium.a-color-base.a-text-normal",
                "span.a-size-base-plus"
            ]
            
            for selector in name_selectors:
                try:
                    name_element = item.find_element(By.CSS_SELECTOR, selector)
                    name = name_element.text.strip()
                    if name and len(name) > 3:
                        print(f"    Found name with selector '{selector}': {name[:60]}")
                        break
                except NoSuchElementException:
                    continue
            
            if not name:
                # Final fallback - try to get name from the link element
                try:
                    link_element = item.find_element(By.CSS_SELECTOR, f"a[href='{product_link}']")
                    name = link_element.get_attribute("aria-label") or link_element.text.strip()
                    print(f"    Found name from link aria-label/text: {name[:60]}")
                except:
                    print(f"    Failed to extract name: No valid name found")
                    continue
            
            if not product_link or not name or len(name) < 3:
                print(f"    Skipping - invalid link or name")
                continue
            
            # Skip accessories only if they're clearly accessories (not the product itself)
            name_lower = name.lower()
            # Only skip if it's CLEARLY an accessory product
            is_accessory = False
            if any(word in name_lower for word in ['back cover', 'phone case', 'mobile cover', 'protective case', 'screen protector', 'tempered glass', 'screen guard']):
                is_accessory = True
            if is_accessory:
                print(f"    Skipping - accessory: {name[:60]}")
                continue
            
            # Validate product matches search query (relaxed for watches)
            # FIX: Don't skip if name is truncated to just brand (e.g., "Apple" instead of "Apple iPhone 15")
            search_tokens_lower = set(product_name.lower().split()) - {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy', 'watch', 'smart', 'smartwatch'}
            if search_tokens_lower and len(search_tokens_lower) > 0:
                brands = ['samsung', 'apple', 'iphone', 'oneplus', 'xiaomi', 'redmi', 'realme', 'oppo', 'vivo', 'pixel', 'galaxy', 'noise', 'titan', 'boat']
                has_brand = any(brand in product_name.lower() for brand in brands)
                if has_brand:
                    # If name is just the brand (truncated), don't skip - full name is on product page
                    is_truncated = name_lower.strip() in brands or len(name_lower.strip()) < 15
                    brand_match = any(term in name_lower for term in search_tokens_lower) or is_truncated
                    if not brand_match:
                        print(f"    Skipping - doesn't match brand: {name[:60]}")
                        continue
            
            # Get basic info from search results
            price_text = "0"
            try:
                price_element = item.find_element(By.CSS_SELECTOR, ".a-price-whole")
                price_text = price_element.text.strip()
            except:
                pass
            
            image_url = ""
            try:
                img_element = item.find_element(By.CSS_SELECTOR, "img.s-image")
                image_url = img_element.get_attribute("src")
            except:
                pass
            
            rating = ""
            try:
                rating_element = item.find_element(By.CSS_SELECTOR, "span.a-icon-alt")
                rating = rating_element.text.strip()
            except:
                pass
            
            reviews = ""
            try:
                reviews_element = item.find_element(By.CSS_SELECTOR, "span.a-size-base.s-underline-text")
                reviews = reviews_element.text.strip()
            except:
                pass
            
            # Open product in new tab
            print(f"    Opening product page in new tab...")
            driver.execute_script("window.open(arguments[0]);", product_link)
            time.sleep(3)
            driver.switch_to.window(driver.window_handles[1])
            print(f"    Switched to new tab, extracting details...")
            
            # Scrape detailed info
            detailed_info = scrape_amazon_product_details(driver, None)
            
            # Close tab and switch back
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            print(f"    Closed tab, back to search results")
            time.sleep(1)
            
            cat, subcat = categorize_product(name)
            product_data = {
                "name": name,
                "price": price_text,
                "price_numeric": clean_price(price_text),
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": cat,
                "subcategory": subcat,
                "source": "Amazon.in",
                "product_link": product_link,
                "availability": "In Stock"
            }
            
            # Add detailed info
            if detailed_info:
                product_data.update(detailed_info)
            
            products.append(product_data)
            print(f"    ✓ Successfully scraped product {idx+1}")
            
        except Exception as e:
            print(f"  ✗ Error on Amazon product {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            # Close any extra tabs and return to main window
            try:
                while len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass
            continue
    
    print(f"Amazon: Successfully scraped {len(products)} products with details")
    return products
def scrape_detailed_flipkart(driver, product_name, max_products=5):
    """Enhanced Flipkart scraper with deep product details"""
    
    # Try up to 2 times if page doesn't load
    for attempt in range(2):
        try:
            if attempt > 0:
                print(f"  Retry attempt {attempt + 1}/2...")
            
            url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
            driver.get(url)
            time.sleep(random.uniform(7, 10))  # Increased wait time
            
            try:
                close_btn = WebDriverWait(driver, 5).until(  # Increased from 3 to 5
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '✕')]"))
                )
                close_btn.click()
                time.sleep(2)  # Increased from 1 to 2
            except:
                pass
            
            # Try to wait for products to load
            WebDriverWait(driver, 30).until(  # Increased from 15 to 30 seconds
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-id], div.CGtC98, div.tUxRFH, div._1AtVbE"))
            )
            time.sleep(5)  # Increased from 3 to 5
            
            # If we get here, page loaded successfully
            break
            
        except TimeoutException:
            if attempt == 0:
                print(f"  Flipkart: First attempt timed out, retrying...")
                continue
            else:
                print(f"  Flipkart: Results took too long to load after 2 attempts (waited 30 seconds each)")
                print(f"  Trying to continue with whatever loaded...")
        except Exception as e:
            print(f"  Flipkart error on attempt {attempt + 1}: {str(e)}")
            if attempt == 0:
                continue
            else:
                return []
    
    products = []
    
    # Find product containers
    item_selectors = ["div[data-id]", "div.CGtC98", "div.tUxRFH", "div._1AtVbE"]
    items = []
    
    for selector in item_selectors:
        items = driver.find_elements(By.CSS_SELECTOR, selector)
        if items:
            print(f"Flipkart: Found {len(items)} items with selector '{selector}', scraping details for top {max_products}...")
            break
    
    if not items:
        print("Flipkart: Could not find product items")
        return []
    
    processed_count = 0
    
    for idx, item in enumerate(items):
        if processed_count >= max_products:
            break
            
        try:
            print(f"  Scraping Flipkart product {processed_count + 1}/{max_products}...")
            
            # Check if product is sold out or unavailable - skip early
            try:
                item_text_lower = item.text.lower()
                if any(skip_text in item_text_lower for skip_text in ['sold out', 'currently unavailable', 'out of stock', 'coming soon']):
                    print(f"    Skipping - product is sold out/unavailable")
                    continue
            except:
                pass
            
            # Get product link and name from link element
            product_link = ""
            name = ""
            
            # Try all link selectors
            link_selectors = [
                "a.wKTcLC",
                "a[href*='/p/']",
                "a.wjcEIp",
                "a.CGtC98",
                "a.rPDeLR",
                "a.VJA3rP",
                "a.IRpwTa"
            ]
            
            for link_sel in link_selectors:
                try:
                    link_elem = item.find_element(By.CSS_SELECTOR, link_sel)
                    href = link_elem.get_attribute("href")
                    if href and ('/p/' in href or '/product/' in href):
                        if '?' in href:
                            href = href.split('?')[0]
                        product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
                        # Try to get name from link title attribute first (most reliable)
                        name = link_elem.get_attribute("title") or link_elem.get_attribute("aria-label") or ""
                        
                        # If no title/aria-label, try text but clean it
                        if not name or len(name) < 10:
                            raw_text = link_elem.text.strip()
                            # Split by newlines and find the actual product name
                            lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
                            for line in lines:
                                # Skip junk lines
                                if "Currently unavailable" in line:
                                    continue
                                if "Add to Compare" in line:
                                    continue
                                if "₹" in line:
                                    continue
                                if "%" in line:
                                    continue
                                if "OFF" in line.upper():
                                    continue
                                if len(line) < 15:
                                    continue
                                name = line
                                break
                        
                        if product_link:
                            print(f"    Found link: {product_link[:60]}...")
                        if name and len(name) > 10:
                            print(f"    Found name from link: {name[:60]}")
                            break
                except:
                    continue
            
            if not product_link:
                print(f"    No valid product link found, skipping...")
                continue
            
            if 'search' in product_link:
                continue
            
            # If name not found or is generic, try other selectors
            generic_names = ['bestseller', 'coming soon', 'new arrival', 'hot deal', 'sale']
            if not name or len(name) < 5 or any(gen in name.lower() for gen in generic_names):
                # First try to get title from anchor tag with title attribute (most reliable)
                try:
                    title_el = item.find_element(By.CSS_SELECTOR, "a[title]")
                    name = title_el.get_attribute("title") or ""
                    if name and len(name) > 15:
                        print(f"    Found name from a[title]: {name[:60]}")
                except:
                    pass
                
                # Try to get all text from the item and extract product name
                if not name or len(name) < 15:
                    try:
                        item_text = item.text
                        lines = [line.strip() for line in item_text.split('\n') if line.strip()]
                        for line in lines:
                            # Skip lines that are clearly not product names
                            if "₹" in line:
                                continue
                            if "%" in line:
                                continue
                            if "OFF" in line.upper():
                                continue
                            if "Add to Compare" in line:
                                continue
                            if "Currently unavailable" in line:
                                continue
                            if len(line) < 15:
                                continue
                            if any(gen in line.lower() for gen in generic_names):
                                continue
                            if line.replace(',', '').replace('(', '').replace(')', '').isdigit():
                                continue
                            if 'rating' in line.lower():
                                continue
                            name = line
                            print(f"    Found name from item text: {name[:60]}")
                            break
                    except:
                        pass
                
                # If still not found, try specific selectors
                if not name or len(name) < 10:
                    name_selectors = [
                        "div.KzDlHZ",
                        "div.wjcEIp",
                        "a.wjcEIp",
                        "div.syl9yP",
                        "a.VJA3rP",
                        "div._2WkVRV",
                        "a.IRpwTa",
                        "div.rPDeLR",
                        "div._4rR01T"
                    ]
                    
                    for sel in name_selectors:
                        try:
                            name_elem = item.find_element(By.CSS_SELECTOR, sel)
                            name = name_elem.text.strip()
                            name = re.sub(r'^\d+\.\s*', '', name)
                            if name and len(name) > 10 and not name.startswith('₹') and not any(gen in name.lower() for gen in generic_names):
                                # Skip junk names
                                if "Add to Compare" in name or "Currently unavailable" in name:
                                    name = ""
                                    continue
                                print(f"    Found name with selector '{sel}': {name[:60]}")
                                break
                        except:
                            continue
            
            # Final validation - skip junk names
            if name and (name.lower() in ["add to compare", "currently unavailable", "out of stock"]):
                print(f"    Skipping junk name: {name}")
                continue
                
            if not name or len(name) < 10 or any(gen in name.lower() for gen in generic_names):
                print(f"    Failed to extract valid name, skipping...")
                continue
            
            # Skip accessories only if they're clearly accessories
            name_lower = name.lower()
            is_accessory = False
            if any(word in name_lower for word in ['back cover', 'phone case', 'mobile cover', 'protective case', 'screen protector', 'tempered glass', 'screen guard']):
                is_accessory = True
            if is_accessory:
                print(f"    Skipping - accessory: {name[:60]}")
                continue
            
            # Validate product matches search query (relaxed for watches)
            # FIX: Don't skip if name is truncated to just brand (e.g., "Apple" instead of "Apple iPhone 15")
            search_tokens_lower = set(product_name.lower().split()) - {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy', 'watch', 'smart', 'smartwatch'}
            if search_tokens_lower and len(search_tokens_lower) > 0:
                brands = ['samsung', 'apple', 'iphone', 'oneplus', 'xiaomi', 'redmi', 'realme', 'oppo', 'vivo', 'pixel', 'galaxy', 'noise', 'titan', 'boat']
                has_brand = any(brand in product_name.lower() for brand in brands)
                if has_brand:
                    # If name is just the brand (truncated), don't skip - full name is on product page
                    is_truncated = name_lower.strip() in brands or len(name_lower.strip()) < 15
                    brand_match = any(term in name_lower for term in search_tokens_lower) or is_truncated
                    if not brand_match:
                        print(f"    Skipping - doesn't match brand: {name[:60]}")
                        continue
            
            # Get price - try multiple selectors
            price_text = "0"
            price_selectors = [
                "div.Nx9bqj",
                "div._30jeq3", 
                "div._1_WHN1",
                "div.hl05eU",
                "div._25b18c"
            ]
            
            for sel in price_selectors:
                try:
                    price_elem = item.find_element(By.CSS_SELECTOR, sel)
                    price_text = price_elem.text.strip()
                    if price_text and '₹' in price_text:
                        print(f"    Found price: {price_text}")
                        break
                except:
                    continue
            
            # Fallback: extract price from item text using regex
            if not price_text or price_text == "0" or '₹' not in price_text:
                try:
                    item_text = item.text
                    price_match = re.search(r'₹\s*[\d,]+', item_text)
                    if price_match:
                        price_text = price_match.group(0)
                        print(f"    Found price from text: {price_text}")
                except:
                    pass
            
            # Get image
            image_url = ""
            try:
                img_elem = item.find_element(By.CSS_SELECTOR, "img")
                image_url = img_elem.get_attribute("src")
            except:
                pass
            
            # Get rating
            rating = ""
            try:
                rating_elem = item.find_element(By.CSS_SELECTOR, "div.XQDdHH, div._3LWZlK, div.Y1HWO0")
                rating = rating_elem.text.strip()
            except:
                pass
            
            # Get reviews
            reviews = ""
            try:
                reviews_elem = item.find_element(By.CSS_SELECTOR, "span._2_R_DZ, span.Wphh3N")
                reviews = reviews_elem.text.strip()
            except:
                pass
            
            # Open product page in new tab
            print(f"    Opening product page in new tab...")
            driver.execute_script("window.open(arguments[0]);", product_link)
            time.sleep(3)
            
            # Switch to new tab
            driver.switch_to.window(driver.window_handles[1])
            print(f"    Switched to new tab, extracting details...")
            
            # Scrape detailed info from product page
            detailed_info = scrape_flipkart_product_details(driver, None)
            
            # Close tab and switch back
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            print(f"    Closed tab, back to search results")
            time.sleep(1)
            
            # If price wasn't found from search results, use price from product page
            if (not price_text or price_text == "0" or '₹' not in price_text) and detailed_info:
                page_price = detailed_info.get('page_price', '')
                if page_price and '₹' in page_price:
                    price_text = page_price
                    print(f"    Using price from product page: {price_text}")
            
            cat, subcat = categorize_product(name)
            product_data = {
                "name": name,
                "price": price_text,
                "price_numeric": clean_price(price_text),
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": cat,
                "subcategory": subcat,
                "source": "Flipkart",
                "product_link": product_link,
                "availability": "In Stock"
            }
            
            # Add detailed info
            if detailed_info:
                product_data.update(detailed_info)
            
            products.append(product_data)
            processed_count += 1
            print(f"    ✓ Successfully scraped product {processed_count}")
            
        except Exception as e:
            print(f"  ✗ Error on Flipkart product {processed_count + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Close any extra tabs and return to main window
            try:
                while len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass
            continue
    
    print(f"Flipkart: Successfully scraped {len(products)} products with details")
    return products

def scrape_detailed_amazon_resilient(driver, product_name, max_products=5):
    """Wrapper around scrape_detailed_amazon to handle Amazon interstitial/503 gracefully."""
    # Visit homepage to clear interstitial if present
    try:
        driver.get("https://www.amazon.in")
        time.sleep(3)
        _ = handle_amazon_interstitial(driver)
    except Exception:
        pass

    # First attempt
    try:
        products = scrape_detailed_amazon(driver, product_name, max_products=max_products)
    except Exception:
        products = []

    # Soft retry: go directly to search URL and refresh once if 503 detected
    if not products:
        try:
            search_url = build_amazon_search_url(product_name)
            driver.get(search_url)
            time.sleep(4)
            blocked = handle_amazon_interstitial(driver)
            if blocked:
                print("Amazon 503/interstitial detected; refreshing search once...")
                driver.get(search_url)
                time.sleep(4)
            products = scrape_detailed_amazon(driver, product_name, max_products=max_products)
        except Exception:
            pass

    return products
def scrape_detailed_flipkart_resilient(driver, product_name, max_products=5):
    """Wrapper around scrape_detailed_flipkart to handle Flipkart interstitial/503 gracefully."""
    # Visit homepage to clear interstitial if present
    try:
        driver.get("https://www.flipkart.com")
        time.sleep(3)
        _ = handle_flipkart_interstitial(driver)
    except Exception:
        pass

    # First attempt
    try:
        products = scrape_detailed_flipkart(driver, product_name, max_products=max_products)
    except Exception:
        products = []

    # Soft retry: go directly to search URL and refresh once if 503 detected
    if not products:
        try:
            search_url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
            driver.get(search_url)
            time.sleep(4)
            blocked = handle_flipkart_interstitial(driver)
            if blocked:
                print("Flipkart 503/interstitial detected; refreshing search once...")
                driver.get(search_url)
                time.sleep(4)
            products = scrape_detailed_flipkart(driver, product_name, max_products=max_products)
        except Exception:
            pass

    return products

# ===========================
# DETAILED CROMA SCRAPER
# ===========================

def scrape_croma_product_details(driver, product_url):
    """Scrape detailed information from Croma product page - using actual Croma selectors"""
    try:
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': [],
            'customer_reviews': [],
            'review_summary': '',
            'rating_breakdown': {},
            'rating': ''
        }
        
        print(f"      Searching for Croma product details...")
        
        # Scroll to load dynamic content
        for _ in range(5):
            driver.execute_script("window.scrollBy(0, 600);")
            time.sleep(0.5)
        
        # ===== KEY FEATURES EXTRACTION =====
        # From Croma: div.key-features-box > ul li
        feature_selectors = [
            "div.key-features-box ul li",
            ".key-features-box li",
            "div.cp-keyfeature ul li",
            ".pd-eligibility-wrap ul li",
            "[class*='key-features'] li",
            "[class*='keyfeature'] li"
        ]
        
        for selector in feature_selectors:
            try:
                items = driver.find_elements(By.CSS_SELECTOR, selector)
                for item in items[:15]:
                    text = item.text.strip()
                    if text and len(text) > 5 and text not in details['features']:
                        # Parse "Display: 6.7 inches..." format
                        if ':' in text:
                            parts = text.split(':', 1)
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                details['technical_details'][key] = value
                        details['features'].append(text)
                if details['features']:
                    print(f"      Found {len(details['features'])} key features from Croma")
                    break
            except:
                continue
        
        # ===== SPECIFICATIONS EXTRACTION =====
        # Try to expand/click specifications section
        try:
            spec_headers = driver.find_elements(By.CSS_SELECTOR, 
                "h2.accordian-title, [class*='accordian-title'], h2[class*='MuiTypography']")
            for header in spec_headers:
                if 'specification' in header.text.lower():
                    header.click()
                    time.sleep(1)
                    break
        except:
            pass
        
        # Extract specs from page text using patterns
        try:
            page_body = driver.find_element(By.TAG_NAME, "body").text
            import re
            spec_patterns = [
                (r'Mobile Type\s*\n?\s*([^\n]+)', 'Mobile Type'),
                (r'Mobile Design\s*\n?\s*([^\n]+)', 'Mobile Design'),
                (r'Brand\s*\n?\s*([^\n]+)', 'Brand'),
                (r'Model Series\s*\n?\s*([^\n]+)', 'Model Series'),
                (r'Model Number\s*\n?\s*([^\n]+)', 'Model Number'),
                (r'Dimensions[^:]*\s*\n?\s*([0-9.x\s]+)', 'Dimensions'),
                (r'Product Weight\s*\n?\s*([^\n]+)', 'Weight'),
                (r'Display[:\s]+([^\n]+)', 'Display'),
                (r'Memory[:\s]+([^\n]+)', 'Memory'),
                (r'Processor[:\s]+([^\n]+)', 'Processor'),
                (r'Camera[:\s]+([^\n]+)', 'Camera'),
                (r'Battery[:\s]+([^\n]+)', 'Battery'),
                (r'Operating System[:\s]+([^\n]+)', 'OS'),
            ]
            for pattern, key in spec_patterns:
                if key not in details['technical_details']:
                    match = re.search(pattern, page_body, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if value and len(value) < 150:
                            details['technical_details'][key] = value
        except:
            pass
        
        print(f"      Found {len(details['technical_details'])} specifications from Croma")
        
        # ===== OVERVIEW/DESCRIPTION EXTRACTION =====
        desc_selectors = [
            "#review_accord h2.accordian-title + div",
            "div[class*='overview'] p",
            ".product-description",
            "[class*='MuiAccordionSummary'] + div p"
        ]
        
        for selector in desc_selectors:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, selector)
                texts = []
                for elem in elems[:3]:
                    text = elem.text.strip()
                    if text and len(text) > 30:
                        texts.append(text)
                if texts:
                    details['description'] = ' '.join(texts)[:2000]
                    break
            except:
                continue
        
        # Combine features into description if empty
        if not details['description'] and details['features']:
            details['description'] = ' | '.join(details['features'])
        
        # ===== RATING EXTRACTION =====
        try:
            rating_selectors = [".cp-rating", "[class*='rating']", ".overall-rating"]
            for sel in rating_selectors:
                try:
                    rating_elem = driver.find_element(By.CSS_SELECTOR, sel)
                    rating_text = rating_elem.text.strip()
                    if rating_text and any(c.isdigit() for c in rating_text):
                        import re
                        match = re.search(r'(\d+\.?\d*)', rating_text)
                        if match:
                            details['rating'] = match.group(1)
                            break
                except:
                    continue
        except:
            pass
        
        # ===== RATING BREAKDOWN =====
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            for star in ['5', '4', '3', '2', '1']:
                pattern = rf'{star}\s*star[:\s]*(\d+)'
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    details['rating_breakdown'][f"{star} Star"] = match.group(1)
        except:
            pass
        
        # ===== CUSTOMER REVIEWS EXTRACTION =====
        try:
            # Look for review containers
            review_text_selectors = [
                "[class*='review-text']",
                "[class*='customer-review'] p",
                ".review-content",
                "[class*='Review']"
            ]
            
            for selector in review_text_selectors:
                try:
                    review_elems = driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in review_elems[:10]:
                        text = elem.text.strip()
                        if text and len(text) > 20:
                            review = {'text': text}
                            # Check for verified
                            if 'verified' in text.lower():
                                review['verified'] = True
                            details['customer_reviews'].append(review)
                    if details['customer_reviews']:
                        break
                except:
                    continue
            
            if details['customer_reviews']:
                print(f"      Found {len(details['customer_reviews'])} customer reviews from Croma")
        except:
            pass
        
        print(f"      Croma extraction complete: {len(details['technical_details'])} specs, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Croma details: {e}")
        return None


def scrape_detailed_croma(driver, product_name, max_products=5):
    """
    Scrape Croma with detailed product information.
    Opens each product page to extract full specs.
    """
    print(f"🔍 Searching Croma for: {product_name}")
    
    products = []
    base_url = "https://www.croma.com"
    
    try:
        search_url = f"{base_url}/search/?q={product_name.replace(' ', '%20')}"
        driver.get(search_url)
        time.sleep(random.uniform(4, 6))
        
        # Wait for products to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    "li.product-item, div.product-card, li.plp-card, div[class*='product']"))
            )
        except TimeoutException:
            print("⚠️ No products found on Croma")
            return products
        
        # Scroll to load more products
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(random.uniform(1, 2))
        
        # Find product elements
        product_selectors = [
            "li.product-item",
            "li.plp-card", 
            "div.product-card",
            "div[class*='cp-product']"
        ]
        
        product_elements = []
        for selector in product_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    product_elements = elements[:max_products + 2]  # Get extra for filtering
                    print(f"✅ Found {len(product_elements)} products on Croma")
                    break
            except:
                continue
        
        if not product_elements:
            print("❌ No product elements found on Croma")
            return products
        
        # Extract basic info and links
        product_links = []
        for elem in product_elements:
            try:
                # Get product link
                link = None
                link_selectors = ["a.product-title", "a[href*='/p/']", "h3 a", "a"]
                for sel in link_selectors:
                    try:
                        link_elem = elem.find_element(By.CSS_SELECTOR, sel)
                        href = link_elem.get_attribute('href')
                        if href and '/p/' in href:
                            link = href if href.startswith('http') else base_url + href
                            break
                    except:
                        continue
                
                # Get name
                name = ""
                name_selectors = ["h3.product-title", ".product-title", "h3 a", "a.product-title"]
                for sel in name_selectors:
                    try:
                        name_elem = elem.find_element(By.CSS_SELECTOR, sel)
                        name = name_elem.text.strip()
                        if name:
                            break
                    except:
                        continue
                
                # Get price
                price_text = ""
                price_numeric = 0
                price_selectors = [".new-price", ".product-price", "span[class*='price']", ".amount"]
                for sel in price_selectors:
                    try:
                        price_elem = elem.find_element(By.CSS_SELECTOR, sel)
                        price_text = price_elem.text.strip()
                        cleaned = re.sub(r'[^\d.]', '', price_text)
                        price_numeric = float(cleaned) if cleaned else 0
                        if price_numeric > 0:
                            break
                    except:
                        continue
                
                # Get image
                image_url = ""
                try:
                    img_elem = elem.find_element(By.CSS_SELECTOR, "img")
                    image_url = img_elem.get_attribute('src') or img_elem.get_attribute('data-src')
                except:
                    pass
                
                # Get rating
                rating = ""
                try:
                    rating_elem = elem.find_element(By.CSS_SELECTOR, "[class*='rating'], [class*='star']")
                    rating = rating_elem.text.strip()
                except:
                    pass
                
                if name and price_numeric > 0:
                    # Skip accessories - same validation as Amazon/Flipkart
                    name_lower = name.lower()
                    accessory_keywords = [
                        'back cover', 'phone case', 'mobile cover', 'protective case', 
                        'screen protector', 'tempered glass', 'screen guard', 'case for',
                        'cover for', 'pouch', 'sleeve', 'skin for', 'charger for',
                        'cable for', 'adapter for', 'holder for', 'stand for'
                    ]
                    is_accessory = any(kw in name_lower for kw in accessory_keywords)
                    if is_accessory:
                        print(f"    Skipping accessory: {name[:50]}")
                        continue
                    
                    # Validate brand match if searching for specific brand
                    brands = ['samsung', 'apple', 'iphone', 'oneplus', 'xiaomi', 'redmi', 'realme', 'oppo', 'vivo', 'pixel', 'galaxy', 'noise', 'titan', 'boat']
                    search_lower = product_name.lower()
                    has_brand_search = any(brand in search_lower for brand in brands)
                    if has_brand_search:
                        # At least one search term should match (brand or model)
                        search_tokens = set(search_lower.split()) - {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy'}
                        is_truncated = len(name_lower.strip()) < 15
                        brand_match = any(term in name_lower for term in search_tokens) or is_truncated
                        if not brand_match:
                            print(f"    Skipping - doesn't match search: {name[:50]}")
                            continue
                    
                    product_links.append({
                        'name': name,
                        'price': price_text,
                        'price_numeric': price_numeric,
                        'image_url': image_url,
                        'product_link': link,
                        'rating': rating,
                        'source': 'Croma'
                    })
            except Exception as e:
                continue
        
        print(f"📦 Found {len(product_links)} valid Croma products")
        
        # Fetch detailed info for each product
        for i, product_data in enumerate(product_links[:max_products]):
            try:
                print(f"  Fetching details for: {product_data['name'][:50]}...")
                
                if product_data.get('product_link'):
                    # Open product in new tab
                    driver.execute_script("window.open(arguments[0]);", product_data['product_link'])
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(random.uniform(2, 4))
                    
                    # Scrape detailed info
                    detailed_info = scrape_croma_product_details(driver, None)
                    
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                    
                    # Add detailed info to product
                    if detailed_info:
                        product_data.update(detailed_info)
                        print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                
                # Add default fields
                product_data.setdefault('category', 'Electronics')
                product_data.setdefault('subcategory', 'General')
                product_data.setdefault('reviews', '')
                product_data.setdefault('technical_details', {})
                product_data.setdefault('features', [])
                product_data.setdefault('description', '')
                
                products.append(product_data)
                
            except Exception as e:
                print(f"    ✗ Error fetching details: {e}")
                # Make sure we're back to main window
                try:
                    while len(driver.window_handles) > 1:
                        driver.switch_to.window(driver.window_handles[-1])
                        driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                except:
                    pass
        
        print(f"✅ Scraped {len(products)} products from Croma with details")
        return products
        
    except Exception as e:
        print(f"❌ Croma scraping failed: {e}")
        return products


# ===========================
# DETAILED RELIANCE DIGITAL SCRAPER
# ===========================

def scrape_reliance_product_details(driver, product_url):
    """Scrape detailed information from Reliance Digital product page"""
    try:
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': [],
            'customer_reviews': [],
            'review_summary': '',
            'rating_breakdown': {}
        }
        
        print(f"      Searching for Reliance Digital product details...")
        
        # Scroll to load dynamic content
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(0.5)
        
        # Extract specifications
        spec_selectors = [
            "div.specifications",
            "div[class*='specification']",
            "table.spec-table",
            "ul.spec-list",
            "div.pdp-specification"
        ]
        
        for selector in spec_selectors:
            try:
                spec_section = driver.find_element(By.CSS_SELECTOR, selector)
                rows = spec_section.find_elements(By.CSS_SELECTOR, "tr, li, div.spec-row, div[class*='spec-item']")
                for row in rows:
                    try:
                        text = row.text.strip()
                        if ':' in text:
                            parts = text.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                if key and value:
                                    details['technical_details'][key] = value
                        elif '\n' in text:
                            lines = text.split('\n')
                            if len(lines) >= 2:
                                key = lines[0].strip()
                                value = lines[1].strip()
                                if key and value:
                                    details['technical_details'][key] = value
                    except:
                        continue
                if details['technical_details']:
                    print(f"      Found {len(details['technical_details'])} specs from Reliance Digital")
                    break
            except:
                continue
        
        # Extract features/highlights
        feature_selectors = [
            "ul.key-features li",
            "div.product-highlights li",
            "ul.highlights li",
            "div[class*='highlight'] li",
            "ul[class*='feature'] li"
        ]
        
        for selector in feature_selectors:
            try:
                features = driver.find_elements(By.CSS_SELECTOR, selector)
                for feat in features[:10]:
                    text = feat.text.strip()
                    if text and len(text) > 5 and text not in details['features']:
                        details['features'].append(text)
                if details['features']:
                    print(f"      Found {len(details['features'])} features from Reliance Digital")
                    break
            except:
                continue
        
        # Extract description
        desc_selectors = [
            "div.product-description",
            "div[class*='description']",
            "div.pdp-desc",
            "p.product-desc"
        ]
        
        for selector in desc_selectors:
            try:
                desc_elem = driver.find_element(By.CSS_SELECTOR, selector)
                text = desc_elem.text.strip()
                if text and len(text) > 50:
                    details['description'] = text[:1000]
                    break
            except:
                continue
        
        # Combine features into description if empty
        if not details['description'] and details['features']:
            details['description'] = ' | '.join(details['features'])
        
        print(f"      Reliance Digital extraction complete: {len(details['technical_details'])} specs, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Reliance Digital details: {e}")
        return None


def scrape_detailed_reliance(driver, product_name, max_products=5):
    """
    Scrape Reliance Digital with detailed product information.
    Opens each product page to extract full specs.
    """
    print(f"🔍 Searching Reliance Digital for: {product_name}")
    
    products = []
    base_url = "https://www.reliancedigital.in"
    
    try:
        # Use full URL format with search_term and internal_source for better results
        encoded_query = product_name.replace(' ', '%20')
        search_url = f"{base_url}/products?q={encoded_query}&search_term={encoded_query}&internal_source=search_prompt&page_no=1&page_size=12&page_type=number"
        print(f"📌 Reliance Digital URL: {search_url}")
        driver.get(search_url)
        time.sleep(random.uniform(4, 6))
        
        # Wait for products to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    ".sp, .product-card, div[class*='product'], li[class*='product'], .pl__container"))
            )
        except TimeoutException:
            print("⚠️ No products found on Reliance Digital")
            return products
        
        # Scroll to load more products
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(random.uniform(1, 2))
        
        # Find product elements
        product_selectors = [
            "div.sp",
            "li.product-card",
            "div.product-card",
            "div[class*='product-item']"
        ]
        
        product_elements = []
        for selector in product_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    product_elements = elements[:max_products + 2]
                    print(f"✅ Found {len(product_elements)} products on Reliance Digital")
                    break
            except:
                continue
        
        if not product_elements:
            print("❌ No product elements found on Reliance Digital")
            return products
        
        # Extract basic info and links
        product_links = []
        for elem in product_elements:
            try:
                # Get product link
                link = None
                try:
                    link_elem = elem.find_element(By.CSS_SELECTOR, "a")
                    href = link_elem.get_attribute('href')
                    if href:
                        link = href if href.startswith('http') else base_url + href
                except:
                    pass
                
                # Get name
                name = ""
                name_selectors = [".sp__name", ".product-title", "h3", "[class*='title']", "a"]
                for sel in name_selectors:
                    try:
                        name_elem = elem.find_element(By.CSS_SELECTOR, sel)
                        name = name_elem.text.strip()
                        if name and len(name) > 5:
                            break
                    except:
                        continue
                
                # Get price
                price_text = ""
                price_numeric = 0
                price_selectors = [".sp__price", ".product-price", "[class*='price']"]
                for sel in price_selectors:
                    try:
                        price_elem = elem.find_element(By.CSS_SELECTOR, sel)
                        price_text = price_elem.text.strip()
                        cleaned = re.sub(r'[^\d.]', '', price_text)
                        price_numeric = float(cleaned) if cleaned else 0
                        if price_numeric > 0:
                            break
                    except:
                        continue
                
                # Get image
                image_url = ""
                try:
                    img_elem = elem.find_element(By.CSS_SELECTOR, "img")
                    image_url = img_elem.get_attribute('src') or img_elem.get_attribute('data-src')
                except:
                    pass
                
                # Get rating
                rating = ""
                try:
                    rating_elem = elem.find_element(By.CSS_SELECTOR, "[class*='rating']")
                    rating = rating_elem.text.strip()
                except:
                    pass
                
                if name and price_numeric > 0:
                    # Skip accessories - same validation as Amazon/Flipkart
                    name_lower = name.lower()
                    accessory_keywords = [
                        'back cover', 'phone case', 'mobile cover', 'protective case', 
                        'screen protector', 'tempered glass', 'screen guard', 'case for',
                        'cover for', 'pouch', 'sleeve', 'skin for', 'charger for',
                        'cable for', 'adapter for', 'holder for', 'stand for'
                    ]
                    is_accessory = any(kw in name_lower for kw in accessory_keywords)
                    if is_accessory:
                        print(f"    Skipping accessory: {name[:50]}")
                        continue
                    
                    # Validate brand match if searching for specific brand
                    brands = ['samsung', 'apple', 'iphone', 'oneplus', 'xiaomi', 'redmi', 'realme', 'oppo', 'vivo', 'pixel', 'galaxy', 'noise', 'titan', 'boat']
                    search_lower = product_name.lower()
                    has_brand_search = any(brand in search_lower for brand in brands)
                    if has_brand_search:
                        # At least one search term should match (brand or model)
                        search_tokens = set(search_lower.split()) - {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy'}
                        is_truncated = len(name_lower.strip()) < 15
                        brand_match = any(term in name_lower for term in search_tokens) or is_truncated
                        if not brand_match:
                            print(f"    Skipping - doesn't match search: {name[:50]}")
                            continue
                    
                    product_links.append({
                        'name': name,
                        'price': price_text,
                        'price_numeric': price_numeric,
                        'image_url': image_url,
                        'product_link': link,
                        'rating': rating,
                        'source': 'Reliance Digital'
                    })
            except Exception as e:
                continue
        
        print(f"📦 Found {len(product_links)} valid Reliance Digital products")
        
        # Fetch detailed info for each product
        for i, product_data in enumerate(product_links[:max_products]):
            try:
                print(f"  Fetching details for: {product_data['name'][:50]}...")
                
                if product_data.get('product_link'):
                    # Open product in new tab
                    driver.execute_script("window.open(arguments[0]);", product_data['product_link'])
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(random.uniform(2, 4))
                    
                    # Scrape detailed info
                    detailed_info = scrape_reliance_product_details(driver, None)
                    
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                    
                    # Add detailed info to product
                    if detailed_info:
                        product_data.update(detailed_info)
                        print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                
                # Add default fields
                product_data.setdefault('category', 'Electronics')
                product_data.setdefault('subcategory', 'General')
                product_data.setdefault('reviews', '')
                product_data.setdefault('technical_details', {})
                product_data.setdefault('features', [])
                product_data.setdefault('description', '')
                
                products.append(product_data)
                
            except Exception as e:
                print(f"    ✗ Error fetching details: {e}")
                # Make sure we're back to main window
                try:
                    while len(driver.window_handles) > 1:
                        driver.switch_to.window(driver.window_handles[-1])
                        driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                except:
                    pass
        
        print(f"✅ Scraped {len(products)} products from Reliance Digital with details")
        return products
        
    except Exception as e:
        print(f"❌ Reliance Digital scraping failed: {e}")
        return products


def scrape_detailed_croma_resilient(driver, product_name, max_products=5):
    """Wrapper around scrape_detailed_croma with retry logic."""
    try:
        products = scrape_detailed_croma(driver, product_name, max_products=max_products)
    except Exception as e:
        print(f"Croma first attempt failed: {e}")
        products = []
    
    if not products:
        try:
            time.sleep(3)
            products = scrape_detailed_croma(driver, product_name, max_products=max_products)
        except Exception:
            pass
    
    return products


def scrape_detailed_reliance_resilient(driver, product_name, max_products=5):
    """Wrapper around scrape_detailed_reliance with retry logic."""
    try:
        products = scrape_detailed_reliance(driver, product_name, max_products=max_products)
    except Exception as e:
        print(f"Reliance Digital first attempt failed: {e}")
        products = []
    
    if not products:
        try:
            time.sleep(3)
            products = scrape_detailed_reliance(driver, product_name, max_products=max_products)
        except Exception:
            pass
    
    return products


def categorize_product(name, subcategory=""):
    """Categorize product based on name with detailed hierarchy"""
    name_lower = (name + " " + subcategory).lower()
    
    categories = {
        'Mobiles, Computers': {
            'keywords': ['phone', 'mobile', 'smartphone', 'iphone', 'galaxy', 'oneplus', 'redmi', 'realme', 'oppo', 'vivo', 'pixel'],
            'subcategories': ['Mi', 'Realme', 'Samsung', 'Apple', 'Vivo', 'OPPO', 'Poco', 'Motorola']
        },
        'Mobile Accessories': {
            'keywords': ['case', 'cover', 'charger', 'cable', 'power bank', 'screenguard', 'tempered glass'],
            'subcategories': ['Mobile Cases', 'Chargers', 'Power Banks', 'Screenguards']
        },
        'Smart Wearable Tech': {
            'keywords': ['smart watch', 'smartwatch', 'fitness band', 'smart band'],
            'subcategories': ['Smart Watches', 'Fitness Bands']
        },
        'Laptops': {
            'keywords': ['laptop', 'notebook', 'macbook'],
            'subcategories': ['Gaming Laptops', 'Business Laptops']
        },
        'Tablets': {
            'keywords': ['tablet', 'ipad'],
            'subcategories': ['Apple iPads', 'Android Tablets']
        },
        'Camera': {
            'keywords': ['camera', 'dslr', 'lens'],
            'subcategories': ['DSLR', 'Mirrorless', 'Lenses']
        },
        'TV, Appliances': {
            'keywords': ['tv', 'television', 'speaker', 'soundbar'],
            'subcategories': ['Televisions', 'Speakers', 'Soundbars']
        },
        'Fashion': {
            'keywords': ['shirt', 'jeans', 'dress', 'shoes'],
            'subcategories': ['Clothing', 'Footwear']
        }
    }
    
    for category, data in categories.items():
        for keyword in data['keywords']:
            if keyword in name_lower:
                for subcat in data['subcategories']:
                    if subcat.lower() in name_lower:
                        return category, subcat
                return category, 'General'
    
    return 'Other', 'Uncategorized'


def clean_price(price_text):
    if not price_text:
        return 0.0
    
    cleaned = re.sub(r'[₹,\s]', '', price_text)
    numbers = re.findall(r'\d+\.?\d*', cleaned)
    
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return 0.0
    return 0.0


def filter_only_phones(products, search_term):
    """Keep only phone-related products - improved to be more lenient"""
    if not products:
        return products

    # Extract meaningful search terms (remove generic words)
    generic_terms = {"mobile", "phone", "phones", "smartphone", "smartphones", "cell", "the", "a", "an", "best"}
    tokens = [t for t in re.split(r'\W+', search_term.lower()) if t and t not in generic_terms]

    # Keywords that indicate it's a phone product
    phone_include = {"phone", "mobile", "smartphone", "iphone", "galaxy", "pixel", "oneplus", "redmi", "realme", "oppo", "vivo"}

    # Only exclude clear accessories/non-phone items
    exclude_keywords = [
        "case", "cover", "bumper", "back cover", "protective case",
        "charger", "cable", "adapter", "power adapter",
        "tempered glass", "screen guard", "screen protector", "glass protector",
        "pouch", "strap", "stand", "holder",
        "battery pack", "lens protector", "camera protector",
        "headphone", "earphone", "earbuds", "airpods",
        "powerbank", "power bank", "charging cable",
        "ring holder", "popsocket", "pop socket"
    ]

    filtered = []
    for p in products:
        title = p.get("name", "").lower()
        if not title:
            continue

        # Check if it's clearly an accessory (must match full phrase for more accuracy)
        is_accessory = False
        for ex in exclude_keywords:
            # Only exclude if the keyword is a clear match
            if ex in title:
                # But don't exclude if it also has phone keywords (e.g., "phone with case")
                if not any(phone_kw in title for phone_kw in phone_include):
                    is_accessory = True
                    break
        
        if is_accessory:
            continue

        # If search has specific tokens (like "iphone", "16"), product must contain them
        if tokens:
            # Check if product name contains the search tokens
            if any(tok in title for tok in tokens):
                filtered.append(p)
                continue
            # Or if it contains general phone keywords
            elif any(f in title for f in phone_include):
                filtered.append(p)
                continue
        else:
            # No specific tokens - keep anything with phone keywords
            if any(f in title for f in phone_include):
                filtered.append(p)

    return filtered


# ===========================
# Report Generation
# ===========================

def create_detailed_report(rag_storage):
    """Create detailed statistics report"""
    stats = rag_storage.get_statistics()
    
    if not stats:
        print("No statistics available")
        return
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PRODUCT DATABASE REPORT".center(70))
    print("="*70)
    print(f"\nTotal Products: {stats['total_products']}")
    print(f"Products with Full Details: {stats.get('detailed_products', 0)}")
    
    print("\n" + "-"*70)
    print("PRODUCTS BY SOURCE")
    print("-"*70)
    for source, count in stats['by_source'].items():
        percentage = (count / stats['total_products']) * 100
        print(f"  {source:20} : {count:4} products ({percentage:.1f}%)")
    
    print("\n" + "-"*70)
    print("PRODUCTS BY CATEGORY")
    print("-"*70)
    for category, count in stats['by_category'].items():
        percentage = (count / stats['total_products']) * 100
        print(f"  {category:20} : {count:4} products ({percentage:.1f}%)")
    
    if stats.get('price_stats'):
        print("\n" + "-"*70)
        print("PRICE STATISTICS")
        print("-"*70)
        ps = stats['price_stats']
        print(f"  Minimum Price      : ₹{ps['min']:,.2f}")
        print(f"  Maximum Price      : ₹{ps['max']:,.2f}")
        print(f"  Average Price      : ₹{ps['avg']:,.2f}")
        print(f"  Median Price       : ₹{ps['median']:,.2f}")
        print(f"  Price Range        : ₹{ps['max'] - ps['min']:,.2f}")
    
    if stats.get('rating_stats'):
        print("\n" + "-"*70)
        print("RATING STATISTICS")
        print("-"*70)
        rs = stats['rating_stats']
        print(f"  Minimum Rating     : {rs['min']:.1f}⭐")
        print(f"  Maximum Rating     : {rs['max']:.1f}⭐")
        print(f"  Average Rating     : {rs['avg']:.2f}⭐")
        print(f"  Median Rating      : {rs['median']:.2f}⭐")
    
    print("\n" + "="*70 + "\n")


# ===========================
# Enhanced GUI
# ===========================

def load_image_from_url(url, size=(130, 130)):
    try:
        response = requests.get(url, timeout=4)
        img = Image.open(BytesIO(response.content))
        img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def load_image_async(img_label, url):
    """Load image in background thread"""
    try:
        img = load_image_from_url(url)
        if img:
            img_label.config(image=img, text="", bg="white")
            img_label.image = img
    except:
        pass


def display_results_gui_with_details(df, rag_storage):
    """Enhanced GUI showing detailed product information"""
    root = tk.Tk()
    root.title("Product Price Comparison - Detailed View")
    root.geometry("1400x800")
    
    # Main frame with scrollbar
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)
    
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Header with price range
    if not df.empty and 'price_numeric' in df.columns:
        min_price = df['price_numeric'].min()
        max_price = df['price_numeric'].max()
        header_text = f"Product Comparison Results - Price Range: ₹{min_price:.0f} - ₹{max_price:.0f}"
    else:
        header_text = "Product Comparison Results"
    
    header = tk.Label(scrollable_frame, text=header_text, font=("Arial", 18, "bold"), pady=10, bg="#2c3e50", fg="white")
    header.pack(fill=tk.X)
    
    # Highlight lowest price
    if not df.empty and 'price_numeric' in df.columns:
        lowest_price = df['price_numeric'].min()
        lowest_info = tk.Label(scrollable_frame, 
                              text=f"✓ LOWEST PRICE: ₹{lowest_price:.0f}", 
                              font=("Arial", 16, "bold"), fg="white", bg="green", pady=8)
        lowest_info.pack(fill=tk.X)
    
    # Display products sorted by price
    for idx, row in df.iterrows():
        # Check if this is the lowest price
        is_lowest = False
        if 'price_numeric' in df.columns:
            is_lowest = (row['price_numeric'] == df['price_numeric'].min())
        
        # Frame color - highlight lowest price
        if is_lowest:
            product_frame = tk.Frame(scrollable_frame, relief=tk.RAISED, 
                                   borderwidth=4, padx=15, pady=15, bg="#e8f5e9")
        else:
            product_frame = tk.Frame(scrollable_frame, relief=tk.RAISED, 
                                   borderwidth=2, padx=15, pady=15, bg="white")
        
        product_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # Image placeholder
        img_label = tk.Label(product_frame, text="Loading...", width=150, height=150, bg="lightgray")
        img_label.grid(row=0, column=0, rowspan=6, padx=10, sticky=tk.N)
        
        # Load image asynchronously
        if row.get('image_url'):
            Thread(target=load_image_async, args=(img_label, row['image_url']), daemon=True).start()
        
        # Product name
        name_label = tk.Label(product_frame, text=row['name'], 
                            font=("Arial", 13, "bold"), wraplength=800, justify=tk.LEFT,
                            bg=product_frame['bg'])
        name_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Price with lowest indicator
        if is_lowest:
            price_text = f"Price: ₹{row['price_numeric']:.0f} ⭐ LOWEST PRICE ⭐"
            price_color = "darkgreen"
            price_font = ("Arial", 16, "bold")
        else:
            price_text = f"Price: ₹{row['price_numeric']:.0f}"
            price_color = "green"
            price_font = ("Arial", 15, "bold")
        
        price_label = tk.Label(product_frame, text=price_text, 
                             font=price_font, fg=price_color, bg=product_frame['bg'])
        price_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Basic info line
        info_parts = [f"Source: {row['source']}", f"Category: {row['category']}"]
        if row.get('rating'):
            info_parts.append(f"Rating: {row['rating']}")
        if row.get('reviews'):
            info_parts.append(f"Reviews: {row['reviews']}")
        
        # Add indicator for detailed specs
        if row.get('technical_details') and len(row.get('technical_details', {})) > 0:
            spec_count = len(row['technical_details'])
            info_parts.append(f"📋 {spec_count} Specs Available")
        
        info_text = " | ".join(info_parts)
        info_label = tk.Label(product_frame, text=info_text, font=("Arial", 10),
                            bg=product_frame['bg'], fg="#555")
        info_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Sentiment display (Neural Network Analysis)
        sentiment = row.get('sentiment', 'unknown')
        sentiment_emoji = row.get('sentiment_emoji', '❓')
        sentiment_score = row.get('sentiment_score', 0.5)
        sentiment_explanation = row.get('sentiment_explanation', '')
        sentiment_source = row.get('sentiment_source', 'description')
        
        sentiment_colors = {
            'positive': '#27ae60',  # Green
            'neutral': '#f39c12',   # Orange  
            'negative': '#e74c3c',  # Red
            'unknown': '#95a5a6'    # Gray
        }
        sentiment_color = sentiment_colors.get(sentiment, '#95a5a6')
        
        # Show source of sentiment (reviews or description)
        source_text = "📝 from reviews" if sentiment_source == "customer_reviews" else "📄 from description"
        
        # Sentiment main label
        sentiment_label = tk.Label(
            product_frame,
            text=f"{sentiment_emoji} Sentiment: {sentiment.upper()} ({sentiment_score:.0%}) {source_text}",
            font=("Arial", 10, "bold"),
            fg=sentiment_color,
            bg=product_frame['bg']
        )
        sentiment_label.grid(row=3, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Sentiment explanation label (why it's positive/negative)
        if sentiment_explanation and sentiment != 'unknown':
            explanation_label = tk.Label(
                product_frame,
                text=f"🧠 {sentiment_explanation}",
                font=("Arial", 9, "italic"),
                fg="#666",
                bg=product_frame['bg'],
                wraplength=700,
                justify=tk.LEFT
            )
            explanation_label.grid(row=4, column=1, sticky=tk.W, padx=10, pady=1)
            next_row = 5
        else:
            next_row = 4
        
        # Description/Features
        if row.get('description'):
            desc_text = row['description'][:200] + "..." if len(row.get('description', '')) > 200 else row.get('description', '')
            desc_label = tk.Label(product_frame, text=f"Description: {desc_text}", 
                                font=("Arial", 9), wraplength=800, justify=tk.LEFT,
                                bg=product_frame['bg'], fg="#333")
            desc_label.grid(row=next_row, column=1, sticky=tk.W, padx=10, pady=3)
            next_row += 1
        
        # Technical Details Button - Show if any detailed data exists
        has_details = (row.get('technical_details') or row.get('additional_info') or 
                      row.get('features') or (row.get('description') and len(row.get('description', '')) > 200))
        
        if has_details:
            def show_details(product_row=row):
                details_window = tk.Toplevel(root)
                details_window.title(f"Details - {product_row['name'][:50]}")
                details_window.geometry("900x700")
                
                # Add scrollbar to details window
                details_frame = ttk.Frame(details_window)
                details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                text_widget = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, font=("Courier", 10))
                text_widget.pack(fill=tk.BOTH, expand=True)
                
                # Header
                details_text = "="*90 + "\n"
                details_text += f"PRODUCT DETAILS - {product_row['source']}\n"
                details_text += "="*90 + "\n\n"
                
                details_text += f"Product Name: {product_row['name']}\n"
                details_text += f"Price: ₹{product_row.get('price_numeric', 0):.0f}\n"
                details_text += f"Source: {product_row['source']}\n"
                details_text += f"Category: {product_row.get('category', 'N/A')}\n"
                
                if product_row.get('rating'):
                    details_text += f"Rating: {product_row['rating']}\n"
                if product_row.get('reviews'):
                    details_text += f"Reviews: {product_row['reviews']}\n"
                if product_row.get('availability'):
                    details_text += f"Availability: {product_row['availability']}\n"
                
                # Neural Network Sentiment Analysis info
                sentiment = product_row.get('sentiment', 'unknown')
                sentiment_score = product_row.get('sentiment_score', 0.5)
                sentiment_emoji = product_row.get('sentiment_emoji', '❓')
                sentiment_explanation = product_row.get('sentiment_explanation', '')
                sentiment_source = product_row.get('sentiment_source', 'description')
                
                details_text += f"\n{sentiment_emoji} NEURAL SENTIMENT ANALYSIS (DistilBERT):\n"
                details_text += f"   Sentiment: {sentiment.upper()}\n"
                details_text += f"   Confidence Score: {sentiment_score:.1%}\n"
                details_text += f"   Source: {'Customer Reviews' if sentiment_source == 'customer_reviews' else 'Product Description'}\n"
                
                if product_row.get('sentiment_confidence'):
                    conf = product_row['sentiment_confidence']
                    if isinstance(conf, dict):
                        details_text += f"   Breakdown: Positive={conf.get('positive', 0):.1%}, "
                        details_text += f"Neutral={conf.get('neutral', 0):.1%}, "
                        details_text += f"Negative={conf.get('negative', 0):.1%}\n"
                    elif isinstance(conf, (int, float)):
                        details_text += f"   Confidence: {conf:.1%}\n"
                
                if sentiment_explanation:
                    details_text += f"\n   🧠 Analysis: {sentiment_explanation}\n"
                
                details_text += "\n" + "="*90 + "\n\n"
                
                # Rating Breakdown (5 star: 61%, etc.)
                if product_row.get('rating_breakdown'):
                    details_text += "⭐ RATING BREAKDOWN:\n" + "-"*90 + "\n"
                    for star, percent in product_row['rating_breakdown'].items():
                        details_text += f"   {star}: {percent}\n"
                    details_text += "\n"
                
                # Customer Reviews
                if product_row.get('customer_reviews'):
                    reviews = product_row['customer_reviews']
                    # Ensure reviews is a list
                    if isinstance(reviews, list) and len(reviews) > 0:
                        details_text += f"💬 CUSTOMER REVIEWS ({len(reviews)} shown):\n" + "-"*90 + "\n"
                        for idx, review in enumerate(reviews[:5], 1):
                            if isinstance(review, dict):
                                if review.get('rating'):
                                    details_text += f"   ⭐ {review['rating']}\n"
                                if review.get('title'):
                                    details_text += f"   📌 {review['title']}\n"
                                if review.get('text'):
                                    review_text = review['text'][:400] + "..." if len(review['text']) > 400 else review['text']
                                    details_text += f"   {review_text}\n"
                            elif isinstance(review, str):
                                details_text += f"   {review[:400]}\n"
                            details_text += "\n"
                
                # Review Summary (Customers say...)
                if product_row.get('review_summary'):
                    details_text += "📊 CUSTOMERS SAY:\n" + "-"*90 + "\n"
                    details_text += f"   {product_row['review_summary']}\n\n"
                
                # Technical Details (Specifications)
                if product_row.get('technical_details') and isinstance(product_row['technical_details'], dict):
                    if len(product_row['technical_details']) > 0:
                        details_text += "📋 SPECIFICATIONS:\n" + "-"*90 + "\n"
                        for key, value in product_row['technical_details'].items():
                            # Handle long values
                            if len(str(value)) > 60:
                                details_text += f"{key}:\n  {value}\n"
                            else:
                                details_text += f"{key:45} : {value}\n"
                        details_text += "\n"
                
                # Additional Information
                if product_row.get('additional_info') and isinstance(product_row['additional_info'], dict):
                    if len(product_row['additional_info']) > 0:
                        details_text += "ℹ️  ADDITIONAL INFORMATION:\n" + "-"*90 + "\n"
                        for key, value in product_row['additional_info'].items():
                            if len(str(value)) > 60:
                                details_text += f"{key}:\n  {value}\n"
                            else:
                                details_text += f"{key:45} : {value}\n"
                        details_text += "\n"
                
                # Features/Highlights
                if product_row.get('features') and isinstance(product_row['features'], list):
                    if len(product_row['features']) > 0:
                        details_text += "✨ FEATURES & HIGHLIGHTS:\n" + "-"*90 + "\n"
                        for idx, feat in enumerate(product_row['features'], 1):
                            details_text += f"{idx}. {feat}\n"
                        details_text += "\n"
                
                # Full Description
                if product_row.get('description'):
                    details_text += "📝 DESCRIPTION:\n" + "-"*90 + "\n"
                    details_text += product_row['description'] + "\n\n"
                
                # Product Link
                if product_row.get('product_link'):
                    details_text += "🔗 PRODUCT LINK:\n" + "-"*90 + "\n"
                    details_text += product_row['product_link'] + "\n\n"
                
                details_text += "="*90 + "\n"
                
                text_widget.insert(tk.END, details_text)
                text_widget.config(state=tk.DISABLED)
                
                # Add a copy button
                def copy_to_clipboard():
                    details_window.clipboard_clear()
                    details_window.clipboard_append(details_text)
                    copy_btn.config(text="✓ Copied!", bg="#27ae60")
                    details_window.after(2000, lambda: copy_btn.config(text="📋 Copy Details", bg="#95a5a6"))
                
                copy_btn = tk.Button(details_window, text="📋 Copy Details", 
                                    command=copy_to_clipboard, bg="#95a5a6", fg="white",
                                    font=("Arial", 10, "bold"), padx=15, pady=5)
                copy_btn.pack(pady=5)
            
            details_btn = tk.Button(product_frame, text="📄 View Full Details", 
                                   command=show_details, bg="#3498db", fg="white",
                                   font=("Arial", 9, "bold"), cursor="hand2", padx=10, pady=3)
            details_btn.grid(row=next_row, column=1, sticky=tk.W, padx=10, pady=5)
            next_row += 1
        
        # Product link
        if row.get('product_link'):
            def open_link(url=row['product_link']):
                webbrowser.open(url)
            
            link_label = tk.Label(product_frame, text="🔗 View on Website", 
                                font=("Arial", 9, "underline"), fg="blue",
                                bg=product_frame['bg'], cursor="hand2")
            link_label.grid(row=next_row, column=1, sticky=tk.W, padx=10, pady=3)
            link_label.bind("<Button-1>", lambda e, url=row['product_link']: webbrowser.open(url))
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    root.mainloop()


# ===========================
# Unified RAG Search Function
# ===========================

def unified_rag_search(product_name, rag_storage, max_products=5):
    """
    Unified RAG-based search with the following strategy:
    1. Search locally first (fast, cached)
    2. Fuzzy match if needed (flexible retrieval)
    3. Fetch externally as last resort (always grows knowledge base)
    4. Always enrich with fresh data (never stale, even cached results get updated)
    5. Use LLM to structure unstructured web data (converts messy snippets to clean schema)
    
    Returns:
        DataFrame with results
    """
    # Start power monitoring
    power_monitor = None
    if POWER_MONITOR_AVAILABLE:
        power_monitor = PowerMonitor()
        power_monitor.start_monitoring()
    
    print(f"\n{'='*70}")
    print(f"🔍 UNIFIED RAG SEARCH".center(70))
    print(f"{'='*70}")
    print(f"Query: {product_name}")
    
    # Step 1: Search locally first (exact match)
    print(f"\n📊 Step 1: Searching local database (exact match)...")
    print("-"*70)
    
    # Extract key search terms from query (remove generic words)
    generic_terms = {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy'}
    search_tokens = set(product_name.lower().split()) - generic_terms
    
    local_results = []
    for item in rag_storage.products:
        item_name = str(item.get('name', '')).lower()
        # Check if ALL key search terms are present in product name
        if search_tokens and all(term in item_name for term in search_tokens):
            local_results.append(item)
    
    if local_results:
        print(f"✓ Found {len(local_results)} exact matches in local database!")
        
        target_count = max_products * 2
        if len(local_results) < target_count:
            print(f"⚠️  Only {len(local_results)} cached, need {target_count}. Scraping more...\n")
        else:
            print(f"✓ Using {len(local_results)} cached products")
            result_df = pd.DataFrame(local_results).sort_values(by="price_numeric")
            return result_df
    
    # Step 2: Fuzzy match if needed (with stricter matching)
    print(f"❌ No exact matches found")
    print(f"\n🔍 Step 2: Fuzzy matching in local database...")
    print("-"*70)
    
    fuzzy_results = []
    for item in rag_storage.products:
        item_name = str(item.get('name', '')).lower()
        # At least 60% of key terms must match for fuzzy search
        if search_tokens:
            match_count = sum(1 for term in search_tokens if term in item_name)
            if match_count >= len(search_tokens) * 0.6:
                fuzzy_results.append(item)
    
    if fuzzy_results:
        print(f"✓ Found {len(fuzzy_results)} fuzzy matches!")
        
        target_count = max_products * 2
        if len(fuzzy_results) < target_count:
            print(f"⚠️  Only {len(fuzzy_results)} cached, need {target_count}. Scraping more...\n")
        else:
            print(f"✓ Using {len(fuzzy_results)} cached products")
            result_df = pd.DataFrame(fuzzy_results).sort_values(by="price_numeric")
            return result_df
    
    # Step 3: Fetch externally as last resort (always grows knowledge base)
    print(f"❌ No fuzzy matches found")
    print(f"\n🌐 Step 3: Fetching fresh data from external sources...")
    print(f"{'='*70}\n")
    
    # Use separate drivers for parallel-like scraping (sequential but organized)
    driver = setup_driver()
    
    amazon_products = []
    flipkart_products = []
    croma_products = []
    reliance_products = []
    
    try:
        print("Scraping Amazon with Full Product Details...")
        print("-"*70)
        amazon_products = scrape_detailed_amazon_resilient(driver, product_name, max_products)
        
        print("\nScraping Flipkart with Full Product Details...")
        print("-"*70)
        flipkart_products = scrape_detailed_flipkart_resilient(driver, product_name, max_products)
        
        print("\nScraping Croma with Full Product Details...")
        print("-"*70)
        croma_products = scrape_detailed_croma_resilient(driver, product_name, max_products)
        
        print("\nScraping Reliance Digital with Full Product Details...")
        print("-"*70)
        reliance_products = scrape_detailed_reliance_resilient(driver, product_name, max_products)
        
    finally:
        driver.quit()

    # Combine all products from all 4 sources
    all_products = amazon_products + flipkart_products + croma_products + reliance_products
    
    print(f"\n📊 Scraped Results Summary:")
    print(f"   Amazon: {len(amazon_products)} products")
    print(f"   Flipkart: {len(flipkart_products)} products")
    print(f"   Croma: {len(croma_products)} products")
    print(f"   Reliance Digital: {len(reliance_products)} products")
    print(f"   Total: {len(all_products)} products")
    
    # Filter 1: Validate products match search query using ProductValidator
    # Brand + Product Line = OK (no strict verification)
    # Series/Model Number = MUST be strictly verified
    print(f"\n🔍 Validating scraped products match search query...")
    
    if PRODUCT_VALIDATOR_AVAILABLE:
        # Use advanced ProductValidator with strict Series/Model verification
        validator = ProductValidator()
        validated_products, rejected_products = validator.filter_products(product_name, all_products)
        removed = len(rejected_products)
        if removed > 0:
            print(f"✓ ProductValidator: Removed {removed} products with wrong Series/Model")
            print(f"   Example: Searching 'Vivo V30' rejects 'Vivo Y100' (V≠Y) and 'Vivo V40' (30≠40)")
    else:
        # Fallback: Basic token matching (less strict)
        validated_products = []
        for p in all_products:
            p_name = str(p.get('name', '')).lower()
            p_desc = str(p.get('description', '')).lower()
            combined_text = p_name + " " + p_desc
            if search_tokens:
                # Check if any search token matches (more lenient)
                match_count = sum(1 for term in search_tokens if term in combined_text)
                if match_count >= 1:  # At least one token must match
                    validated_products.append(p)
            else:
                validated_products.append(p)
        
        removed = len(all_products) - len(validated_products)
        if removed > 0:
            print(f"✓ Basic filter: Removed {removed} irrelevant products, kept {len(validated_products)} matching")
    
    # If too many removed, scrape more to reach target (from all 4 sources)
    target_count = max_products * 4  # 4 sources now
    if removed > 0 and len(validated_products) < target_count:
        shortage = max(2, (target_count - len(validated_products)) // 4)
        print(f"⚠️  Need more products. Re-scraping {shortage} per source...")
        
        driver = setup_driver()
        try:
            extra_amazon = scrape_detailed_amazon_resilient(driver, product_name, shortage)
            extra_flipkart = scrape_detailed_flipkart_resilient(driver, product_name, shortage)
            extra_croma = scrape_detailed_croma_resilient(driver, product_name, shortage)
            extra_reliance = scrape_detailed_reliance_resilient(driver, product_name, shortage)
        finally:
            driver.quit()
        
        extra_products = extra_amazon + extra_flipkart + extra_croma + extra_reliance
        
        # Use ProductValidator for extra products too
        if PRODUCT_VALIDATOR_AVAILABLE:
            extra_validated, _ = validator.filter_products(product_name, extra_products)
            validated_products.extend(extra_validated)
            print(f"✓ Added {len(extra_validated)} more validated products from re-scraping")
        else:
            for p in extra_products:
                p_name = str(p.get('name', '')).lower()
                p_desc = str(p.get('description', '')).lower()
                combined_text = p_name + " " + p_desc
                if search_tokens:
                    match_count = sum(1 for term in search_tokens if term in combined_text)
                    if match_count >= 1:
                        validated_products.append(p)
            print(f"✓ Added {len(extra_products)} more products from re-scraping")
    
    all_products = validated_products
    
    # Filter 2: Apply phone filter if searching for phones
    phone_search_terms = ['phone', 'mobile', 'smartphone', 'iphone', 'galaxy', 'pixel', 'oneplus', 'redmi']
    is_phone_search = any(term in product_name.lower() for term in phone_search_terms)
    
    if is_phone_search:
        before_count = len(all_products)
        all_products = filter_only_phones(all_products, product_name)
        after_count = len(all_products)
        if before_count != after_count:
            print(f"📱 Phone Filter: Removed {before_count - after_count} accessories, kept {after_count} phone products")

    if not all_products:
        print("\n❌ No products found after filtering.")
        return None
    
    # Analyze sentiment for all products using Neural Network
    if NEURAL_SENTIMENT_AVAILABLE:
        try:
            neural_analyzer = NeuralSentimentAnalyzer()
            if neural_analyzer.is_ready:
                print(f"\n🧠 Analyzing product sentiment using Neural Network (DistilBERT)...")
                all_products = neural_analyzer.analyze_products_batch(all_products)
            else:
                print("⚠️ Neural model not ready")
                for p in all_products:
                    p['sentiment'] = 'unknown'
                    p['sentiment_score'] = 0.5
                    p['sentiment_emoji'] = '❓'
        except Exception as e:
            print(f"⚠️ Neural sentiment analysis error: {e}")
            for p in all_products:
                p['sentiment'] = 'unknown'
                p['sentiment_score'] = 0.5
                p['sentiment_emoji'] = '❓'
    else:
        for p in all_products:
            p['sentiment'] = 'unknown'
            p['sentiment_score'] = 0.5
            p['sentiment_emoji'] = '❓'
    
    # Products already have complete data from scraping
    print(f"\n✓ Scraped {len(all_products)} products with complete details")

    print(f"\n{'='*70}")
    print("Storing Products in RAG Database (growing knowledge base)...")
    print("-"*70)
    rag_storage.add_products_batch(all_products)
    print(f"✓ Successfully stored {len(all_products)} products with full details")
    print(f"✓ Auto-saved to database: {rag_storage.storage_file}")
    
    print(f"\n{'='*70}")
    print("Generating Analysis Report...")
    print("-"*70)
    create_detailed_report(rag_storage)
    
    # === UMAP Clustering Visualization ===
    if UMAP_ANALYZER_AVAILABLE and len(all_products) >= 10:
        print(f"\n{'='*70}")
        print("🗺️  UMAP Clustering Visualization...")
        print("-"*70)
        try:
            umap_analyzer = UMAPAnalyzer(all_products)
            umap_analyzer.prepare_features()
            umap_analyzer.run_umap()
            metrics = umap_analyzer.calculate_clustering_metrics()
            
            # Save visualization
            from datetime import datetime
            umap_file = f"umap_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            umap_analyzer.create_visualization(umap_file)
            
            print(f"✓ UMAP visualization saved: {umap_file}")
            print(f"   Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
            print(f"   Davies-Bouldin Index: {metrics.get('davies_bouldin_index', 'N/A'):.4f}")
            print(f"   Cluster Purity: {metrics.get('cluster_purity', 'N/A'):.1f}%")
        except Exception as e:
            print(f"⚠️ UMAP visualization failed: {e}")
    elif not UMAP_ANALYZER_AVAILABLE:
        print("\n⚠️ UMAP not available. Install with: pip install umap-learn matplotlib")
    
    df = pd.DataFrame(all_products)
    
    # Filter out products with 0 price before sorting
    df_valid_price = df[df['price_numeric'] > 0]
    if not df_valid_price.empty:
        df = df_valid_price
    
    df = df.sort_values(by="price_numeric")
    
    print(f"{'='*70}")
    print("UNIFIED RAG SEARCH COMPLETED!".center(70))
    print(f"{'='*70}")
    print(f"\n📊 Total Products Found: {len(df)}")
    if not df.empty and 'price_numeric' in df.columns:
        valid_prices = df[df['price_numeric'] > 0]
        if not valid_prices.empty:
            print(f"💰 Lowest Price: ₹{valid_prices['price_numeric'].min():.0f}")
            print(f"💰 Highest Price: ₹{valid_prices['price_numeric'].max():.0f}")
    detailed_count = sum(1 for p in all_products if p.get('technical_details') or p.get('additional_info'))
    print(f"📝 Products with Full Details: {detailed_count}/{len(df)}")
    
    # Show source breakdown
    if 'source' in df.columns:
        print(f"\n📦 By Source:")
        for source in df['source'].unique():
            count = len(df[df['source'] == source])
            print(f"   {source}: {count} products")
    
    # === Power Consumption Report ===
    if power_monitor:
        print(f"\n{'='*70}")
        print("⚡ Power Consumption Report...")
        print("-"*70)
        try:
            power_monitor.record_measurement("Scraping Complete")
            report = power_monitor.generate_report()
            # Extract from nested structure
            cpu_usage = report.get('resource_utilization', {}).get('average_cpu_usage_percent', 0)
            cpu_power = report.get('power_consumption', {}).get('average_cpu_power_watts', 0)
            mem_usage = report.get('resource_utilization', {}).get('average_memory_usage_percent', 0)
            total_energy = report.get('energy_consumption', {}).get('total_energy_kwh', 0)
            co2_grams = report.get('co2_emissions_grams', {}).get('india', 0)
            duration = report.get('summary', {}).get('total_duration_seconds', 0)
            
            print(f"   CPU Usage (avg): {cpu_usage:.1f}%")
            print(f"   CPU Power (avg): {cpu_power:.2f}W")
            print(f"   Memory Usage (avg): {mem_usage:.1f}%")
            print(f"   Total Energy: {total_energy:.6f} kWh")
            print(f"   CO₂ Emissions: {co2_grams/1000:.6f} kg")
            print(f"   Duration: {duration:.1f}s")
        except Exception as e:
            print(f"   ⚠️ Power report error: {e}")
    
    print(f"\n{'='*70}\n")
    
    return df


# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    # Initialize RAG storage
    rag_storage = ProductRAGStorage('product_rag_database.pkl')
    
    print("\n" + "="*70)
    print("🛍️  E-COMMERCE PRICE COMPARISON WITH RAG".center(70))
    print("="*70)
    
    while True:
        print("\n📋 Main Menu:")
        print("1. 🔍 Search Products (Unified RAG: Local → Fuzzy → Web Scraping)")
        print("2. 📊 View Database Statistics")
        print("3. 🗑️  Clear Database")
        print("4. 💾 Export Database to CSV")
        print("5. 💡 Semantic Search in DB")
        print("6. 📜 List All Products")
        print("7. ❌ Delete a Product by ID")
        print("8. ℹ️  View Product Details by ID")
        print("9. 🚪 Exit")
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            # Unified RAG search
            product_name = input("\n🔍 Enter product name to search: ").strip()
            if product_name:
                max_products = input("📦 How many products per source? (default: 5): ").strip()
                max_products = int(max_products) if max_products.isdigit() else 5
                
                result_df = unified_rag_search(product_name, rag_storage, max_products)
                
                if result_df is not None and not result_df.empty:
                    print("\n🖥️  Opening GUI with detailed product information...")
                    display_results_gui_with_details(result_df, rag_storage)
                else:
                    print("\n❌ No products found.")
        
        elif choice == "2":
            # Statistics
            create_detailed_report(rag_storage)
        
        elif choice == "3":
            # Clear database
            confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ").strip().lower()
            if confirm == "yes":
                rag_storage.clear_storage()
                print("✓ Database cleared successfully.")
        
        elif choice == "4":
            filename = input("📁 Enter CSV filename (default: products_export.csv): ").strip()
            filename = filename or "products_export.csv"
            rag_storage.export_to_csv(filename)
            
        elif choice == "5":
            # Semantic Search in DB
            query = input("\n💡 Enter search query for semantic search: ").strip()
            if query:
                results = rag_storage.semantic_search(query)
                if results:
                    print(f"\nFound {len(results)} semantic matches:")
                    for p in results:
                        print(f"  - [Score: {p['similarity_score']:.2f}] {p['name']} (ID: {p['id']})")
                else:
                    print("No semantic matches found in the database.")

        elif choice == "6":
            # List All Products
            if rag_storage.products:
                print("\n📜 All products in database:")
                for p in rag_storage.products:
                    print(f"  - ID: {p['id']}, Name: {p['name']}")
            else:
                print("Database is empty.")

        elif choice == "7":
            # Delete a Product
            product_id = input("\n❌ Enter the ID of the product to delete: ").strip()
            if product_id:
                if rag_storage.delete_product(product_id):
                    print(f"✓ Product with ID '{product_id}' deleted.")
                else:
                    print(f"✗ Product with ID '{product_id}' not found.")

        elif choice == "8":
            # View Product Details
            product_id = input("\nℹ️  Enter the ID of the product to view: ").strip()
            if product_id:
                product = rag_storage.get_product_by_id(product_id)
                if product:
                    print("\n" + "="*70)
                    print(f"DETAILS FOR PRODUCT: {product.get('id')}".center(70))
                    print("="*70)
                    for key, value in product.items():
                        if isinstance(value, dict) and value:
                            print(f"\n{key.replace('_', ' ').title()}:")
                            for k, v in value.items():
                                print(f"  - {k}: {v}")
                        elif isinstance(value, list) and value:
                            print(f"\n{key.replace('_', ' ').title()}:")
                            for item in value:
                                print(f"  - {item}")
                        else:
                            print(f"{key.replace('_', ' ').title()}: {value}")
                    print("="*70)
                else:
                    print(f"✗ Product with ID '{product_id}' not found.")
            
        elif choice == "9":
            # Exit
            print("\n👋 Exiting... Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-9.")

