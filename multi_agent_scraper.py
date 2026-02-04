from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import time
import random
import re
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO
from threading import Thread
import webbrowser

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

# Additional imports for enhanced features
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import string
import pickle
import os
import concurrent.futures
import numpy as np


# ==========================================================
# ADVANCED REVIEW SCRAPER
# ==========================================================

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
        """Extract Flipkart reviews with AJAX handling"""
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
                self.human_like_scroll(scrolls=8)
            
            reviews_extracted = 0
            page_num = 1
            
            while reviews_extracted < max_reviews and page_num <= 5:
                print(f"📄 Scraping page {page_num}...")
                
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
                    break
                
                for elem in review_elements:
                    if reviews_extracted >= max_reviews:
                        break
                    
                    try:
                        review_data = self._extract_flipkart_review_data(elem)
                        if review_data and review_data['text']:
                            all_reviews.append(review_data)
                            reviews_extracted += 1
                    except:
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
            'verified': False
        }
        
        try:
            text_selectors = ["div[class*='t-ZTKy']", "div.t-ZTKy", "div[class*='_6K-7Co']"]
            for selector in text_selectors:
                try:
                    text_elem = element.find_element(By.CSS_SELECTOR, selector)
                    review['text'] = text_elem.text.strip()
                    if review['text']:
                        break
                except:
                    continue
            
            try:
                rating_elem = element.find_element(By.CSS_SELECTOR, "div[class*='_3LWZlK'], div.XQDdHH")
                rating_text = rating_elem.text.strip()
                review['rating'] = float(rating_text.split()[0]) if rating_text else 0
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
        except:
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


# ==========================================================
# CROMA SCRAPER
# ==========================================================

class CromaScraper:
    """Scraper for Croma.com with detailed product page extraction"""
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 15)
        self.base_url = "https://www.croma.com"
    
    def search_products(self, query, max_products=10):
        """Search for products on Croma with detailed info"""
        print(f"🔍 Searching Croma for: {query}")
        
        products = []
        
        try:
            # Updated URL format: /searchB with relevance sorting
            encoded_query = query.replace(' ', '%20')
            search_url = f"{self.base_url}/searchB?q={encoded_query}%3Arelevance&text={encoded_query}"
            self.driver.get(search_url)
            time.sleep(random.uniform(3, 5))
            
            try:
                self.wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".product-item, .product, li.plp-card, div[class*='product']")
                ))
            except TimeoutException:
                print("⚠️ No products found on Croma")
                return products
            
            for _ in range(3):
                self.driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(random.uniform(1, 2))
            
            product_selectors = [".product-item", "li.plp-card", ".cp-product", "div[class*='product']"]
            
            product_elements = []
            for selector in product_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        product_elements = elements[:max_products + 2]
                        print(f"✅ Found {len(product_elements)} products on Croma")
                        break
                except:
                    continue
            
            # First pass: extract basic info and links
            product_links = []
            for elem in product_elements:
                try:
                    product = self._extract_croma_product(elem)
                    if product and product['name'] and product['price_numeric'] > 0:
                        product_links.append(product)
                except Exception as e:
                    continue
            
            print(f"📦 Found {len(product_links)} valid Croma products")
            
            # Second pass: fetch detailed info from product pages
            for i, product_data in enumerate(product_links[:max_products]):
                try:
                    if product_data.get('product_link'):
                        print(f"  Fetching details for: {product_data['name'][:50]}...")
                        
                        # Open product in new tab
                        self.driver.execute_script("window.open(arguments[0]);", product_data['product_link'])
                        time.sleep(2)
                        self.driver.switch_to.window(self.driver.window_handles[1])
                        time.sleep(random.uniform(2, 4))
                        
                        # Scrape detailed info
                        detailed_info = self._scrape_product_details()
                        
                        # Close tab and switch back
                        self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[0])
                        time.sleep(1)
                        
                        if detailed_info:
                            product_data.update(detailed_info)
                            print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                    
                    products.append(product_data)
                    
                except Exception as e:
                    print(f"    ✗ Error fetching details: {e}")
                    # Make sure we're back to main window
                    try:
                        while len(self.driver.window_handles) > 1:
                            self.driver.switch_to.window(self.driver.window_handles[-1])
                            self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[0])
                    except:
                        pass
                    # Still add the product with basic info
                    products.append(product_data)
            
            print(f"✅ Extracted {len(products)} products from Croma with details")
            return products
            
        except Exception as e:
            print(f"❌ Croma scraping failed: {e}")
            return products
    
    def _scrape_product_details(self):
        """Scrape detailed info from Croma product page - using actual Croma selectors"""
        details = {
            'technical_details': {},
            'features': [],
            'description': '',
            'customer_reviews': [],
            'rating': '',
            'rating_breakdown': {}
        }
        
        try:
            time.sleep(random.uniform(2, 3))
            
            # Scroll to load all content sections
            for _ in range(5):
                self.driver.execute_script("window.scrollBy(0, 600);")
                time.sleep(0.5)
            
            # ===== KEY FEATURES EXTRACTION =====
            # From screenshot: div.key-features-box > h2.feature-text + ul li
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
                    items = self.driver.find_elements(By.CSS_SELECTOR, selector)
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
                        print(f"      Found {len(details['features'])} key features")
                        break
                except:
                    continue
            
            # ===== SPECIFICATIONS EXTRACTION =====
            # From screenshot: Sections like "MOBILE CATEGORY", "MANUFACTURER DETAILS", "PRODUCT DIMENSIONS"
            # Structure: h3/heading for category, then key-value pairs
            
            # Try to expand/click specifications section
            try:
                spec_headers = self.driver.find_elements(By.CSS_SELECTOR, 
                    "h2.accordian-title, [class*='accordian-title'], h2[class*='MuiTypography']")
                for header in spec_headers:
                    if 'specification' in header.text.lower():
                        header.click()
                        time.sleep(1)
                        break
            except:
                pass
            
            # Extract specs from accordion/container sections
            spec_container_selectors = [
                ".cp-section.accordContainer",
                "[class*='specification']",
                ".container .sec-cont",
                "[id*='specification']"
            ]
            
            for container_sel in spec_container_selectors:
                try:
                    containers = self.driver.find_elements(By.CSS_SELECTOR, container_sel)
                    for container in containers:
                        # Look for key-value pairs in the container
                        # Try flex/grid layout items
                        text = container.text
                        lines = text.split('\n')
                        i = 0
                        while i < len(lines) - 1:
                            key = lines[i].strip()
                            value = lines[i+1].strip() if i+1 < len(lines) else ''
                            # Skip section headers and navigation
                            if key and value and len(key) < 50 and len(value) < 200:
                                if key.lower() not in ['specifications', 'overview', 'reviews', 'view more', 'view less']:
                                    if not key.isupper() or len(key) < 30:  # Skip pure headers
                                        details['technical_details'][key] = value
                            i += 2
                    if len(details['technical_details']) >= 5:
                        break
                except:
                    continue
            
            # Also try direct div pairs (label + value)
            try:
                page_body = self.driver.find_element(By.TAG_NAME, "body").text
                # Extract specific patterns
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
            
            print(f"      Found {len(details['technical_details'])} specifications")
            
            # ===== OVERVIEW/DESCRIPTION EXTRACTION =====
            # From screenshot: h2 "Overview" with description paragraphs
            desc_selectors = [
                "#review_accord h2.accordian-title + div",
                "div[class*='overview'] p",
                ".product-description",
                "[class*='MuiAccordionSummary'] + div p"
            ]
            
            for selector in desc_selectors:
                try:
                    elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
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
            
            # ===== REVIEWS EXTRACTION =====
            # From screenshot: Reviews section with rating breakdown and customer reviews
            # Rating breakdown: "5 star: 2", "4 star: 1", etc.
            # Customer reviews with name, date, rating, text
            
            # Get overall rating
            try:
                rating_selectors = [".cp-rating", "[class*='rating']", ".overall-rating"]
                for sel in rating_selectors:
                    try:
                        rating_elem = self.driver.find_element(By.CSS_SELECTOR, sel)
                        rating_text = rating_elem.text.strip()
                        if rating_text and any(c.isdigit() for c in rating_text):
                            # Extract just the number like "2.8"
                            import re
                            match = re.search(r'(\d+\.?\d*)', rating_text)
                            if match:
                                details['rating'] = match.group(1)
                                break
                    except:
                        continue
            except:
                pass
            
            # Get rating breakdown (5 star, 4 star, etc.)
            try:
                breakdown_container = self.driver.find_elements(By.CSS_SELECTOR, 
                    "[class*='rating'] a, .star-rating a, [class*='star']")
                for elem in breakdown_container:
                    text = elem.text.strip().lower()
                    for star in ['5 star', '4 star', '3 star', '2 star', '1 star']:
                        if star in text:
                            # Try to find the count next to it
                            parent = elem.find_element(By.XPATH, "..")
                            parent_text = parent.text
                            import re
                            match = re.search(rf'{star[0]}\s*star[:\s]*(\d+)', parent_text, re.IGNORECASE)
                            if match:
                                details['rating_breakdown'][f"{star[0]} Star"] = match.group(1)
            except:
                pass
            
            # Get customer reviews
            try:
                review_containers = self.driver.find_elements(By.CSS_SELECTOR,
                    "[class*='review'], [class*='Review'], .customer-review")
                
                for container in review_containers[:10]:
                    try:
                        review = {}
                        text = container.text.strip()
                        
                        # Look for verified purchase indicator
                        if 'verified' in text.lower():
                            review['verified'] = True
                        
                        # Try to parse review structure
                        lines = text.split('\n')
                        if len(lines) >= 2:
                            # First line often has name/rating
                            # Look for rating stars or number
                            for line in lines:
                                if '★' in line or 'star' in line.lower():
                                    review['rating'] = line.strip()
                                elif len(line) > 50:  # Likely review text
                                    review['text'] = line.strip()
                            
                            if review.get('text'):
                                details['customer_reviews'].append(review)
                    except:
                        continue
                
                if details['customer_reviews']:
                    print(f"      Found {len(details['customer_reviews'])} customer reviews")
            except:
                pass
            
            return details
            
        except Exception as e:
            print(f"    ⚠️ Error in Croma details extraction: {e}")
            return details
    
    def _extract_croma_product(self, element):
        """Extract product data from Croma listing"""
        product = {
            'name': '',
            'price': '',
            'price_numeric': 0,
            'rating': '',
            'reviews': '',
            'image_url': '',
            'product_link': '',
            'source': 'Croma',
            'category': 'Electronics',
            'subcategory': 'General',
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
            
            price_selectors = [".product-price .amount", ".new-price", "span[class*='price']"]
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
                img_elem = element.find_element(By.CSS_SELECTOR, "img")
                product['image_url'] = img_elem.get_attribute('src') or img_elem.get_attribute('data-src')
            except:
                pass
            
            return product
        except:
            return product
    
    def _clean_price(self, price_text):
        try:
            cleaned = re.sub(r'[^\d.]', '', price_text)
            return float(cleaned) if cleaned else 0
        except:
            return 0


# ==========================================================
# RELIANCE DIGITAL SCRAPER
# ==========================================================

class RelianceDigitalScraper:
    """Scraper for RelianceDigital.in with detailed product page extraction"""
    
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
        """Search for products on Reliance Digital with detailed product page scraping"""
        print(f"🔍 Searching Reliance Digital for: {query}")
        
        products = []
        query_variations = self._generate_search_variations(query)
        product_elements = []
        successful_variation = query
        
        for variation in query_variations:
            try:
                # Use full URL format with search_term and internal_source for better results
                encoded_query = variation.replace(' ', '%20')
                search_url = f"{self.base_url}/products?q={encoded_query}&search_term={encoded_query}&internal_source=search_prompt&page_no=1&page_size=12&page_type=number"
                print(f"📌 Trying Reliance Digital URL: {search_url}")
                self.driver.get(search_url)
                time.sleep(random.uniform(3, 5))
                
                # Dismiss notification popup if present
                try:
                    no_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'No, don')]")
                    no_btn.click()
                    time.sleep(0.5)
                except:
                    pass
                
                try:
                    self.wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, ".sp, .product-card, div[class*='product'], li[class*='product'], .pl__container")
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
                for selector in product_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            product_elements = elements[:max_products * 3]
                            successful_variation = variation
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
        
        # Extract basic info and product links
        basic_products = []
        for elem in product_elements:
            try:
                product = self._extract_reliance_product(elem)
                # Check relevance - at least one keyword should match
                if product and product['name']:
                    name_lower = product['name'].lower()
                    is_relevant = any(word in name_lower for word in query_words if len(word) > 2)
                    
                    if is_relevant:
                        basic_products.append(product)
                        print(f"  ✓ Extracted: {product['name'][:50]}... Link: {bool(product.get('product_link'))}")
                        if len(basic_products) >= max_products:
                            break
                    else:
                        print(f"  ⚠️ Skipping irrelevant: {product['name'][:40]}...")
            except Exception as e:
                print(f"  ⚠️ Failed to extract Reliance product: {e}")
                continue
        
        # Now go into each product page to get detailed specifications
        # Skip if we already have enough info (name, price, image)
        needs_details = [p for p in basic_products if not p.get('technical_details')]
        
        if needs_details:
            print(f"📄 Getting detailed specs from {len(needs_details)} Reliance product pages...")
            main_window = self.driver.current_window_handle
            
            for i, product in enumerate(needs_details):
                if product.get('product_link'):
                    try:
                        print(f"  📖 Opening Reliance product {i+1}/{len(needs_details)}: {product['name'][:40]}...")
                        
                        # Open product page in new tab
                        self.driver.execute_script(f"window.open('{product['product_link']}', '_blank');")
                        time.sleep(random.uniform(1.5, 2.5))
                        
                        # Switch to new tab
                        windows = self.driver.window_handles
                        if len(windows) > 1:
                            self.driver.switch_to.window(windows[-1])
                            
                            # Get detailed info from product page
                            detailed_info = self._scrape_product_details()
                            
                            # Merge detailed info
                            if detailed_info.get('technical_details'):
                                product['technical_details'] = detailed_info['technical_details']
                            if detailed_info.get('features'):
                                product['features'] = detailed_info['features']
                            if detailed_info.get('description'):
                                product['description'] = detailed_info['description']
                            if detailed_info.get('rating'):
                                product['rating'] = detailed_info['rating']
                            if detailed_info.get('reviews'):
                                product['reviews'] = detailed_info['reviews']
                            if detailed_info.get('image_url') and not product.get('image_url'):
                                product['image_url'] = detailed_info['image_url']
                            
                            # Close product tab
                            self.driver.close()
                            self.driver.switch_to.window(main_window)
                            time.sleep(random.uniform(0.3, 0.6))
                    except Exception as e:
                        print(f"    ⚠️ Error getting Reliance product details: {e}")
                        try:
                            self.driver.switch_to.window(main_window)
                        except:
                            pass
                
                products.append(product)
        else:
            products = basic_products
        
        print(f"✅ Extracted {len(products)} detailed products from Reliance Digital")
        return products
    
    def _scrape_product_details(self):
        """Scrape detailed specifications from a Reliance Digital product page"""
        details = {
            'technical_details': {},
            'features': [],
            'description': '',
            'rating': '',
            'reviews': '',
            'image_url': ''
        }
        
        try:
            time.sleep(random.uniform(2, 3))
            
            # Scroll to load all content
            for _ in range(4):
                self.driver.execute_script("window.scrollBy(0, 600);")
                time.sleep(0.5)
            
            # Try to click on "Specifications" tab if exists
            try:
                spec_tabs = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='specifications-tab'], button[contains(text(),'Specification')], .tab-specifications, [class*='spec-tab']")
                for tab in spec_tabs:
                    try:
                        tab.click()
                        time.sleep(1)
                        break
                    except:
                        continue
            except:
                pass
            
            # Extract specifications from tables - try all tables on page
            spec_selectors = [
                ".pdp__specification table",
                ".specifications table",
                ".spec-table",
                "[class*='specification'] table",
                ".product-specs table",
                "table.specs",
                "#specifications table",
                "table",
                "[class*='Specification'] table"
            ]
            
            for selector in spec_selectors:
                try:
                    tables = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for table in tables:
                        rows = table.find_elements(By.CSS_SELECTOR, "tr")
                        for row in rows:
                            try:
                                cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                                if len(cells) >= 2:
                                    key = cells[0].text.strip()
                                    value = cells[1].text.strip()
                                    if key and value and len(key) < 100 and len(value) < 500:
                                        # Skip navigation/menu items
                                        if key.lower() not in ['home', 'products', 'category', 'brand']:
                                            details['technical_details'][key] = value
                            except:
                                continue
                    if len(details['technical_details']) >= 3:
                        break
                except:
                    continue
            
            # Also try to extract from page text patterns if tables didn't work
            if len(details['technical_details']) < 3:
                try:
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text
                    import re
                    patterns = [
                        (r'RAM[:\s]+(\d+\s*GB)', 'RAM'),
                        (r'Internal Storage[:\s]+(\d+\s*GB)', 'Storage'),
                        (r'Display[:\s]+([^\n]{5,50})', 'Display'),
                        (r'Battery[:\s]+(\d+\s*mAh)', 'Battery'),
                        (r'Processor[:\s]+([^\n]{5,50})', 'Processor'),
                        (r'Camera[:\s]+([^\n]{5,50})', 'Camera'),
                        (r'Operating System[:\s]+([^\n]{5,30})', 'OS'),
                    ]
                    for pattern, key in patterns:
                        match = re.search(pattern, page_text, re.IGNORECASE)
                        if match:
                            details['technical_details'][key] = match.group(1).strip()
                except:
                    pass
            
            # Extract from spec key-value pairs
            kv_selectors = [
                ".pdp__specification .spec-row",
                ".specification-row",
                "[class*='spec-item']",
                ".product-spec-row",
                "[class*='specRow']",
                "li[class*='spec']"
            ]
            
            for selector in kv_selectors:
                try:
                    items = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for item in items:
                        try:
                            # Try direct key-value elements
                            key_elem = item.find_element(By.CSS_SELECTOR, ".spec-key, .spec-label, [class*='key'], [class*='label']")
                            value_elem = item.find_element(By.CSS_SELECTOR, ".spec-value, [class*='value']")
                            key = key_elem.text.strip()
                            value = value_elem.text.strip()
                            if key and value:
                                details['technical_details'][key] = value
                        except:
                            # Try parsing text
                            text = item.text.strip()
                            if ':' in text:
                                parts = text.split(':', 1)
                                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                                    details['technical_details'][parts[0].strip()] = parts[1].strip()
                            continue
                except:
                    continue
            
            # Extract features/highlights
            feature_selectors = [
                ".pdp__highlights li",
                ".product-highlights li",
                ".key-features li",
                "[class*='highlight'] li",
                ".feature-list li",
                ".pdp__keyfeatures li",
                "[class*='KeyFeature'] li",
                "[class*='feature'] li"
            ]
            
            for selector in feature_selectors:
                try:
                    items = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for item in items:
                        text = item.text.strip()
                        if text and len(text) > 5 and text not in details['features']:
                            details['features'].append(text)
                except:
                    continue
            
            # Extract description
            desc_selectors = [
                ".pdp__description",
                ".product-description",
                "[class*='description']",
                ".pdp__overview",
                "[class*='ProductDetail']"
            ]
            
            for selector in desc_selectors:
                try:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    desc = elem.text.strip()
                    if desc and len(desc) > 20:
                        details['description'] = desc[:2000]
                        break
                except:
                    continue
            
            # Extract rating
            rating_selectors = [
                ".pdp__rating span",
                ".rating-value",
                "[class*='rating'] span",
                ".star-rating"
            ]
            
            for selector in rating_selectors:
                try:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    rating = elem.text.strip()
                    if rating and any(c.isdigit() for c in rating):
                        details['rating'] = rating
                        break
                except:
                    continue
            
            # Extract reviews count
            review_selectors = [
                ".pdp__reviews",
                "[class*='review-count']",
                ".rating-count"
            ]
            
            for selector in review_selectors:
                try:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    reviews = elem.text.strip()
                    if reviews:
                        details['reviews'] = reviews
                        break
                except:
                    continue
            
            # Get main product image
            img_selectors = [
                ".pdp__image img",
                ".product-image img",
                ".main-image img",
                "[class*='product-img'] img"
            ]
            
            for selector in img_selectors:
                try:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    src = elem.get_attribute('src') or elem.get_attribute('data-src')
                    if src and src.startswith('http'):
                        details['image_url'] = src
                        break
                except:
                    continue
            
            print(f"    ✅ Got {len(details['technical_details'])} specs, {len(details['features'])} features from Reliance page")
            
        except Exception as e:
            print(f"    ⚠️ Error extracting Reliance product details: {e}")
        
        return details
    
    def _extract_reliance_product(self, element):
        """Extract product data from Reliance Digital listing"""
        product = {
            'name': '',
            'price': '',
            'price_numeric': 0,
            'rating': '',
            'reviews': '',
            'image_url': '',
            'product_link': '',
            'source': 'Reliance Digital',
            'category': 'Electronics',
            'subcategory': 'General',
            'technical_details': {},
            'features': [],
            'description': ''
        }
        
        try:
            # Get full text first for filtering and fallback extraction
            full_text = element.text.strip()
            
            # Updated name selectors based on current site structure
            name_selectors = ["p[class*='name']", "a p", ".sp__name", ".product-title", "h3", "[class*='title']", ".pl__title", ".pl__name"]
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
            
            # Try multiple approaches to get product link
            link_selectors = ["a[href*='/products/']", "a[href*='/p/']", "a.product-link", "a"]
            for selector in link_selectors:
                try:
                    link_elem = element.find_element(By.CSS_SELECTOR, selector)
                    href = link_elem.get_attribute('href')
                    if href:
                        product['product_link'] = href if href.startswith('http') else self.base_url + href
                        break
                except:
                    continue
            
            # Updated price selectors - look for ₹ symbol
            price_selectors = ["span[class*='price']", "span[class*='Price']", ".sp__price", "[class*='amount']", ".pl__amount"]
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
                price_match = re.search(r'₹[\d,]+\.?\d*', full_text)
                if price_match:
                    product['price'] = price_match.group()
                    product['price_numeric'] = self._clean_price(price_match.group())
            
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
        except:
            return product
    
    def _clean_price(self, price_text):
        try:
            cleaned = re.sub(r'[^\d.]', '', price_text)
            return float(cleaned) if cleaned else 0
        except:
            return 0


# ==========================================================
# STEALTH BROWSER (Anti-Detection)
# ==========================================================

class StealthBrowser:
    """Anti-detection browser automation"""
    
    def __init__(self, use_proxy=False, proxy_list=None):
        self.options = webdriver.ChromeOptions()
        self.use_proxy = use_proxy
        self.proxy_list = proxy_list or []
        self._configure_stealth()
        self.driver = None
        self.session_file = 'browser_session.pkl'
    
    def start(self):
        """Start the stealth browser"""
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.options
        )
        self._apply_stealth_scripts()
        return self.driver
    
    def _configure_stealth(self):
        """Configure Chrome options for stealth"""
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        resolutions = ['1920,1080', '1366,768', '1440,900']
        self.options.add_argument(f'--window-size={random.choice(resolutions)}')
        
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--lang=en-IN')
        
        if self.use_proxy and self.proxy_list:
            proxy = random.choice(self.proxy_list)
            self.options.add_argument(f'--proxy-server={proxy}')
    
    def _apply_stealth_scripts(self):
        """Apply JavaScript to hide automation"""
        if not self.driver:
            return
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        self.driver.execute_script("window.navigator.chrome = {runtime: {}};")
    
    def human_like_scroll(self, scrolls=3):
        """Scroll like a human"""
        if not self.driver:
            return
        for _ in range(scrolls):
            distance = random.randint(200, 800)
            self.driver.execute_script(f"window.scrollBy(0, {distance});")
            time.sleep(random.uniform(0.5, 1.5))
    
    def smart_wait(self, min_seconds=2, max_seconds=5):
        """Wait with random delays"""
        time.sleep(random.uniform(min_seconds, max_seconds))
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()


# ==========================================================
# RATE LIMITER
# ==========================================================

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


# ==========================================================
# ENHANCED SENTIMENT ANALYZER (Multi-Model)
# ==========================================================

class EnhancedSentimentAnalyzer:
    """Multi-model sentiment analyzer with aspect-based analysis"""
    
    def __init__(self, use_gpu=True):
        self.device = -1
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
            'quality': ['quality', 'build', 'material', 'durable', 'sturdy'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'smooth'],
            'value': ['worth', 'price', 'expensive', 'cheap', 'value', 'money'],
            'battery': ['battery', 'charge', 'charging', 'power', 'backup'],
            'camera': ['camera', 'photo', 'picture', 'image', 'video'],
            'display': ['screen', 'display', 'resolution', 'brightness']
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load sentiment analysis models"""
        if not NEURAL_SENTIMENT_AVAILABLE:
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
    
    def analyze_product(self, reviews_list, product_name="", description=""):
        """Comprehensive product sentiment analysis"""
        if not reviews_list and not description:
            return self._empty_analysis()
        
        all_texts = []
        if description:
            all_texts.append(description[:500])
        
        review_texts = []
        for review in reviews_list:
            if isinstance(review, dict):
                text = review.get('text', '')
                if text:
                    review_texts.append(text)
                    all_texts.append(text)
        
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
        
        ensemble_scores = self._ensemble_sentiment(all_texts[:20])
        results['overall_sentiment'] = ensemble_scores['label']
        results['confidence'] = ensemble_scores['score']
        results['sentiment_distribution'] = ensemble_scores['distribution']
        
        if review_texts:
            results['aspect_sentiments'] = self._aspect_sentiment_analysis(review_texts)
        
        results['key_phrases'] = self._extract_key_phrases(review_texts)
        results['review_breakdown'] = self._categorize_reviews(reviews_list)
        results['recommendation_score'] = self._calculate_recommendation_score(results)
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _ensemble_sentiment(self, texts):
        """Combine predictions from models"""
        scores = {'positive': [], 'negative': [], 'neutral': []}
        
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
                    else:
                        scores['negative'].append(score)
                except:
                    pass
        
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
                    if any(keyword in sentence.lower() for keyword in keywords):
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
        
        positive_words = ['excellent', 'great', 'amazing', 'perfect', 'love', 'best']
        negative_words = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful']
        
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
        """Categorize reviews by rating"""
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
        """Calculate recommendation score (0-100)"""
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
        
        summary = f"Overall Sentiment: {sentiment} ({confidence:.1%} confidence). "
        
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


# ==========================================================
# UNIFIED PRODUCT SCRAPER
# ==========================================================

class UnifiedProductScraper:
    """Master scraper that coordinates all platforms and analysis"""
    
    def __init__(self, use_gpu=True):
        print("🚀 Initializing Unified Product Scraper...")
        
        self.driver = self._setup_driver()
        self.review_scraper = AdvancedReviewScraper(self.driver)
        self.croma = CromaScraper(self.driver)
        self.reliance = RelianceDigitalScraper(self.driver)
        self.rate_limiter = RateLimiter(requests_per_minute=8)
        
        if NEURAL_SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = EnhancedSentimentAnalyzer(use_gpu=use_gpu)
        else:
            self.sentiment_analyzer = None
        
        print("✅ Unified scraper ready!")
    
    def _setup_driver(self):
        """Setup Chrome driver with anti-detection measures"""
        stealth = StealthBrowser()
        return stealth.start()
    
    def search_all_platforms(self, query, max_per_platform=5):
        """Search across all platforms"""
        print(f"\n🔍 Searching ALL platforms for: '{query}'")
        print("="*60)
        
        results = {
            'amazon': [],
            'flipkart': [],
            'croma': [],
            'reliance_digital': [],
            'timestamp': time.time(),
            'query': query
        }
        
        # Rate limit check
        self.rate_limiter.wait_if_needed()
        
        # Search Croma
        try:
            results['croma'] = self.croma.search_products(query, max_per_platform)
        except Exception as e:
            print(f"❌ Croma failed: {e}")
        
        # Rate limit check
        self.rate_limiter.wait_if_needed()
        
        # Search Reliance Digital
        try:
            results['reliance_digital'] = self.reliance.search_products(query, max_per_platform)
        except Exception as e:
            print(f"❌ Reliance Digital failed: {e}")
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate comprehensive comparison report"""
        print("\n📊 Generating Comparison Report...")
        
        all_products = []
        for platform, products in results.items():
            if platform not in ['timestamp', 'query']:
                all_products.extend(products)
        
        if not all_products:
            return {'error': 'No products found'}
        
        report = {
            'total_products': len(all_products),
            'platforms_searched': sum(1 for k, v in results.items() 
                                     if k not in ['timestamp', 'query'] and v),
            'price_analysis': self._analyze_prices(all_products),
            'best_deals': self._find_best_deals(all_products)
        }
        
        return report
    
    def _analyze_prices(self, products):
        """Analyze price distribution"""
        prices = [p.get('price_numeric', 0) for p in products if p.get('price_numeric', 0) > 0]
        
        if not prices:
            return {}
        
        return {
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices),
            'range': max(prices) - min(prices)
        }
    
    def _find_best_deals(self, products):
        """Find best deals across platforms"""
        if not products:
            return []
        
        sorted_products = sorted(
            [p for p in products if p.get('price_numeric', 0) > 0],
            key=lambda x: x.get('price_numeric', float('inf'))
        )
        
        return sorted_products[:5]
    
    def close(self):
        """Cleanup"""
        try:
            self.driver.quit()
            print("✅ Browser closed")
        except:
            pass


# ==========================================================
# GLOBAL CONFIG
# ==========================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


def setup_driver():
    """Create a Selenium Chrome driver with anti-bot tweaks."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation", "enable-logging"]
    )
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_experimental_option(
        "prefs", {"profile.default_content_setting_values.notifications": 2}
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )

    # Avoid basic bot detection
    try:
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride",
            {"userAgent": random.choice(USER_AGENTS)},
        )
    except Exception:
        pass

    try:
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    except Exception:
        pass

    driver.set_page_load_timeout(60)
    return driver


def categorize_product(name, subcategory=""):
    """Very simple category tagging just for display."""
    text = (name + " " + subcategory).lower()

    shoe_keywords = [
        "shoe",
        "sneaker",
        "boot",
        "sandal",
        "slipper",
        "footwear",
        "nike air",
        "jordan",
        "running",
    ]
    clothing_keywords = [
        "shirt",
        "t-shirt",
        "pant",
        "jean",
        "jacket",
        "hoodie",
        "dress",
        "skirt",
        "shorts",
        "clothing",
        "apparel",
    ]

    if any(k in text for k in shoe_keywords):
        return "Shoes"
    if any(k in text for k in clothing_keywords):
        return "Clothing"
    return "Other"


# ==========================================================
# PRODUCT DETAIL SCRAPERS
# ==========================================================

def scrape_amazon_product_details(driver, product_url):
    """Scrape detailed information from Amazon product page - enhanced version"""
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
                # Alternative: Look for insights section
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
                    
                    # Review title
                    try:
                        title = review.find_element(By.CSS_SELECTOR, "[data-hook='review-title'], .review-title").text.strip()
                        review_data['title'] = title
                    except:
                        pass
                    
                    # Review text
                    try:
                        body = review.find_element(By.CSS_SELECTOR, "[data-hook='review-body'], .review-text, .review-text-content").text.strip()
                        review_data['text'] = body
                    except:
                        pass
                    
                    # Review rating
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
        
        # Alternative: Get reviews from review section
        if not details['customer_reviews']:
            try:
                review_texts = driver.find_elements(By.CSS_SELECTOR, ".reviewText, [data-hook='review-body'] span")[:5]
                for rt in review_texts:
                    text = rt.text.strip()
                    if text and len(text) > 20:
                        details['customer_reviews'].append({'text': text})
            except:
                pass
        
        # ========== ORIGINAL DETAIL EXTRACTION ==========
        # Keywords to search for in page elements
        detail_keywords = ['detail', 'product information', 'specification', 'technical', 'product details']
        
        # Method 1: Technical Details table by ID
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
            print(f"      Technical Details table not found by ID, trying alternatives...")
        
        # Method 2: Search all tables for keyword matches
        if not details['technical_details']:
            try:
                all_tables = driver.find_elements(By.TAG_NAME, "table")
                print(f"      Found {len(all_tables)} tables, searching for details...")
                
                for table in all_tables:
                    try:
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
        
        # Additional Information section
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
        
        # Detail bullets (common format on Amazon)
        if not details['technical_details']:
            try:
                bullets = driver.find_elements(By.CSS_SELECTOR, "#detailBullets_feature_div li")
                for bullet in bullets:
                    text = bullet.text.strip()
                    if ':' in text:
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                details['technical_details'][key] = value
            except:
                pass
        
        # Product Features (bullet points)
        desc_found = False
        try:
            desc_element = driver.find_element(By.ID, "feature-bullets")
            features = desc_element.find_elements(By.TAG_NAME, "li")
            print(f"      Found {len(features)} product features")
            for feat in features:
                text = feat.text.strip()
                if text and len(text) > 5:
                    details['features'].append(text)
            if details['features']:
                details['description'] = ' | '.join(details['features'])
                desc_found = True
        except:
            pass
        
        # Alternative description from productDescription div
        if not desc_found:
            try:
                desc = driver.find_element(By.ID, "productDescription")
                details['description'] = desc.text.strip()
                print(f"      Found product description")
                desc_found = True
            except:
                pass
        
        # Method 3: Search divs for description keywords
        if not desc_found:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    try:
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
        
        # Fallback: aplus description
        if not details['description']:
            try:
                aplus = driver.find_element(By.ID, "aplus")
                details['description'] = aplus.text.strip()[:500]
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
                        # Extract just the numeric rating like "4.6"
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
            # Look for the rating bars section
            rating_section = driver.find_elements(By.CSS_SELECTOR, "div.KAdfFz, div._1YokD2, [class*='rating']")
            
            # Extract star breakdown from page text
            page_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            for star in ['5', '4', '3', '2', '1']:
                # Pattern: "5 ★ 2,12,393" or "5★ 2,12,393"
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
            # From screenshot: boxes showing 4.6 Camera, 4.2 Battery, etc.
            category_boxes = driver.find_elements(By.CSS_SELECTOR, 
                "div._2d4LTz, div.vdNlyb, [class*='category-rating']")
            
            page_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            categories = ['Camera', 'Battery', 'Display', 'Design', 'Performance', 'Value']
            for cat in categories:
                # Pattern: "4.6\nCamera" or "4.6 Camera"
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
                "div.xdON2G",  # Specifications container
                "div.GNDEQ-",
                "div._14cfVK",
                "div[class*='specification']"
            ]
            
            for container_sel in spec_section_selectors:
                try:
                    containers = driver.find_elements(By.CSS_SELECTOR, container_sel)
                    for container in containers:
                        # Get section title (e.g., "Other Details", "General")
                        section_title = "Specifications"
                        try:
                            title_elem = container.find_element(By.CSS_SELECTOR, 
                                "div.d2eoIN, div._4BJ2V\\+, [class*='title']")
                            section_title = title_elem.text.strip()[:50] or "Specifications"
                        except:
                            pass
                        
                        # Extract table rows
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
        # From screenshot: div.qFfOgN "Highlights" with bullet points
        try:
            highlight_selectors = [
                "div._1mXcCf li",  # Highlights list
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
        # From screenshot: div.KgDEGp "Product Description" with paragraphs
        try:
            desc_selectors = [
                "div.KgDEGp",  # Product Description container
                "div.RmoJUa",  # Description text
                "div._1mXcCf",
                "div[class*='description']"
            ]
            
            for sel in desc_selectors:
                try:
                    desc_elem = driver.find_element(By.CSS_SELECTOR, sel)
                    text = desc_elem.text.strip()
                    if text and len(text) > 50:
                        # Clean up the description
                        if 'Product Description' in text:
                            text = text.replace('Product Description', '').strip()
                        details['description'] = text[:2000]
                        print(f"      Found product description ({len(details['description'])} chars)")
                        break
                except:
                    continue
        except:
            pass
        
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


# ==========================================================
# SCRAPERS
# ==========================================================

def scrape_amazon_in(driver, product_name, max_products=5, fetch_details=True):
    """
    Scrape Amazon.in search results.
    
    Args:
        fetch_details: If True, open each product page for detailed info (slower).
                      If False, only scrape search page data (faster).
    """
    try:
        print("Opening Amazon...")
        driver.get("https://www.amazon.in")
        time.sleep(random.uniform(4, 6))

        url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
        print("Amazon search URL:", url)
        driver.get(url)
        time.sleep(random.uniform(5, 7))

        WebDriverWait(driver, 35).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-component-type='s-search-result']")
            )
        )
        time.sleep(3)
    except TimeoutException:
        print("Amazon: results took too long to load.")
        return []
    except Exception as e:
        print(f"Amazon error: {e}")
        return []

    products = []
    items = driver.find_elements(
        By.CSS_SELECTOR, "[data-component-type='s-search-result']"
    )

    print(f"Amazon: located {len(items)} search result containers, scraping top {max_products}...")

    # Skip top 2 results (usually ads), then take max_products
    items = items[2:2+max_products]

    for item in items:
        try:
            # Product Link
            product_link = ""
            for sel in ["h2 a", "a.a-link-normal[href*='/dp/']", "a[href*='/dp/']"]:
                try:
                    link_el = item.find_element(By.CSS_SELECTOR, sel)
                    href = link_el.get_attribute("href")
                    if href and '/dp/' in href:
                        product_link = href
                        break
                except NoSuchElementException:
                    continue

            # Name
            name_el = None
            for sel in ["h2 a span", "h2 span"]:
                try:
                    name_el = item.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue
            if not name_el:
                continue
            name = name_el.text.strip()
            if not name:
                continue

            # Price
            price_el = None
            for sel in [".a-price-whole", ".a-price .a-offscreen"]:
                try:
                    price_el = item.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue
            if not price_el:
                continue
            price_text = price_el.text.strip()
            if not price_text:
                continue

            # Image
            image_url = ""
            try:
                img_el = item.find_element(By.CSS_SELECTOR, "img.s-image")
                image_url = img_el.get_attribute("src")
            except Exception:
                pass

            # Rating
            rating = ""
            try:
                r_el = item.find_element(By.CSS_SELECTOR, ".a-icon-alt")
                rating = r_el.get_attribute("textContent").strip()
            except Exception:
                pass

            # Reviews
            reviews = ""
            try:
                rev_el = item.find_element(
                    By.CSS_SELECTOR, ".a-size-base.s-underline-text"
                )
                reviews = rev_el.text.strip()
            except Exception:
                pass

            # Subcategory
            subcategory = ""
            try:
                sub_el = item.find_element(
                    By.CSS_SELECTOR, ".a-size-base-plus.a-color-base"
                )
                subcategory = sub_el.text.strip()
            except Exception:
                pass

            category = categorize_product(name, subcategory)

            product_data = {
                "name": name,
                "subcategory": subcategory,
                "price": price_text,
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": category,
                "source": "Amazon.in",
                "product_link": product_link,
                "technical_details": {},
                "features": [],
                "description": "",
            }
            
            # Fetch detailed info from product page (only if fetch_details is True)
            if fetch_details and product_link:
                try:
                    print(f"  Fetching details for: {name[:50]}...")
                    # Open product in new tab
                    driver.execute_script("window.open(arguments[0]);", product_link)
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(random.uniform(2, 4))
                    
                    # Scrape detailed info
                    detailed_info = scrape_amazon_product_details(driver, None)
                    
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                    
                    # Add detailed info to product
                    if detailed_info:
                        product_data.update(detailed_info)
                        print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                    
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
            
            products.append(product_data)

        except Exception:
            continue

    print(f"Found {len(products)} products on Amazon")
    return products


def scrape_flipkart(driver, product_name, max_products=5, fetch_details=True):
    """
    Scrape Flipkart search results.
    
    Args:
        fetch_details: If True, open each product page for detailed info (slower).
                      If False, only scrape search page data (faster).
    """
    """
    More tolerant Flipkart scraper:
    - uses sleep + multiple selector fallbacks
    - falls back to parsing text (name/price) from item.text
    """
    try:
        url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
        print("Opening Flipkart search URL:", url)
        driver.get(url)
        time.sleep(random.uniform(7, 9))

        # Close login popup if present
        try:
            close_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), '✕')]")
                )
            )
            close_btn.click()
            time.sleep(1)
        except Exception:
            pass

        time.sleep(random.uniform(5, 7))

    except TimeoutException:
        print("Flipkart: page load timed out.")
        return []
    except Exception as e:
        print(f"Flipkart error while loading page: {e}")
        return []

    products = []
    items = []

    possible_selectors = [
        "._1AtVbE",
        "._13oc-S",
        "._tUxRFH",
        "._CGtC98",
        "div[data-id]",  # generic product tiles
    ]

    for sel in possible_selectors:
        items = driver.find_elements(By.CSS_SELECTOR, sel)
        if items:
            print(f"Flipkart: found {len(items)} containers, scraping top {max_products}...")
            break

    if not items:
        print("Flipkart: no product containers found.")
        return []

    for item in items[:max_products]:
        try:
            full_text = item.text.strip()
            if not full_text:
                continue

            # Skip sold out / unavailable products early
            full_text_lower = full_text.lower()
            if any(skip in full_text_lower for skip in ['sold out', 'currently unavailable', 'out of stock', 'coming soon']):
                print(f"  Skipping unavailable product")
                continue

            # ---------------- Product Link ----------------
            product_link = ""
            try:
                link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                href = link_el.get_attribute("href")
                if href:
                    product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
            except NoSuchElementException:
                pass
            
            if not product_link:
                try:
                    link_el = item.find_element(By.CSS_SELECTOR, "a[title]")
                    href = link_el.get_attribute("href")
                    if href:
                        product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
                except NoSuchElementException:
                    pass

            # ---------------- Name ----------------
            name = ""
            
            # First, try to get title from anchor tag (most reliable)
            try:
                title_el = item.find_element(By.CSS_SELECTOR, "a[title]")
                name = title_el.get_attribute("title") or ""
            except NoSuchElementException:
                pass
            
            # Try specific class selectors
            if not name:
                for sel in ["._4rR01T", ".KzDlHZ", ".s1Q9rs", ".WKTcLC", "._2WkVRV"]:
                    try:
                        name_el = item.find_element(By.CSS_SELECTOR, sel)
                        name = name_el.text.strip()
                        if name and len(name) > 15:  # Valid product name
                            break
                    except NoSuchElementException:
                        continue
            
            # Fallback: parse from full text, skip junk lines
            if not name or len(name) < 15:
                lines = [l.strip() for l in full_text.split("\n") if l.strip()]
                for line in lines:
                    # Skip junk
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
                    # Skip very short lines (likely ratings like "4.5")
                    if len(line) < 15:
                        continue
                    name = line
                    break

            if not name or len(name) < 10:
                continue
            
            # Skip if the name is obviously not a product
            if name.lower() in ["add to compare", "currently unavailable", "out of stock"]:
                continue

            # ---------------- Price ----------------
            price_el = None
            for sel in ["._30jeq3", "._1_WHN1", "._Nx9bqj", "._4b5DiR"]:
                try:
                    price_el = item.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue

            price_text = ""
            if price_el:
                price_text = price_el.text.strip()

            if not price_text:
                m = re.search(r"₹\s*[\d,]+", full_text)
                if m:
                    price_text = m.group(0)

            # Note: Don't skip here if no price - we'll try to get it from product page later

            # ---------------- Image ----------------
            image_url = ""
            try:
                img_el = item.find_element(By.CSS_SELECTOR, "img")
                image_url = img_el.get_attribute("src")
            except Exception:
                pass

            # ---------------- Rating ----------------
            rating = ""
            try:
                r_el = item.find_element(By.CSS_SELECTOR, "div[class*='_3LWZlK']")
                rating = r_el.text.strip()
            except Exception:
                pass

            # ---------------- Reviews ----------------
            reviews = ""
            try:
                rev_el = item.find_element(
                    By.CSS_SELECTOR, "span[class*='_2_R_DZ']"
                )
                reviews = rev_el.text.strip()
            except Exception:
                pass

            # ---------------- Subcategory / brand text ----------------
            subcategory = ""
            try:
                sub_el = item.find_element(
                    By.CSS_SELECTOR, "._2WkVRV, ._NqpwHC"
                )
                subcategory = sub_el.text.strip()
            except Exception:
                pass

            category = categorize_product(name, subcategory)

            product_data = {
                "name": name,
                "subcategory": subcategory,
                "price": price_text,
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": category,
                "source": "Flipkart",
                "product_link": product_link,
                "technical_details": {},
                "features": [],
                "description": "",
            }
            
            # Fetch detailed info from product page (only if fetch_details is True)
            if fetch_details and product_link:
                try:
                    print(f"  Fetching details for: {name[:50]}...")
                    # Open product in new tab
                    driver.execute_script("window.open(arguments[0]);", product_link)
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(random.uniform(2, 4))
                    
                    # Scrape detailed info
                    detailed_info = scrape_flipkart_product_details(driver, None)
                    
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                    
                    # Add detailed info to product
                    if detailed_info:
                        product_data.update(detailed_info)
                        # Use page_price as fallback if no price from search page
                        if not product_data.get('price') and detailed_info.get('page_price'):
                            product_data['price'] = detailed_info['page_price']
                            print(f"    → Using price from product page: {detailed_info['page_price']}")
                        print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                    
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
            
            # Skip product if still no price (only strict when not fetching details)
            if not product_data.get('price'):
                if fetch_details:
                    print(f"  Skipping product without price: {name[:50]}...")
                    continue
                # In fast mode, keep products even without price from search page
            
            products.append(product_data)

        except Exception:
            continue

    print(f"Found {len(products)} products on Flipkart")
    return products


# ==========================================================
# FILTERING + GUI HELPERS
# ==========================================================

def clean_price(price_text: str) -> float:
    if not price_text:
        return 0.0
    cleaned = re.sub(r"[₹,\s]", "", price_text)
    nums = re.findall(r"\d+\.?\d*", cleaned)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            return 0.0
    return 0.0


def load_image_from_url(url, size=(130, 130)):
    try:
        response = requests.get(url, timeout=4)
        img = Image.open(BytesIO(response.content))
        img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def load_image_async(img_label, url):
    try:
        img = load_image_from_url(url)
        if img:
            img_label.config(image=img, text="", bg="white")
            img_label.image = img
    except Exception:
        pass


def display_results_gui(df: pd.DataFrame):
    root = tk.Tk()
    root.title("Product Price Comparison")
    root.geometry("1200x800")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Enable mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    header = tk.Label(
        scrollable_frame,
        text="Product Price Comparison Results",
        font=("Arial", 18, "bold"),
        pady=10,
    )
    header.pack()

    for _, row in df.iterrows():
        frame = tk.Frame(
            scrollable_frame, relief=tk.RAISED, borderwidth=2, padx=10, pady=10
        )
        frame.pack(fill=tk.X, padx=10, pady=5)

        img_label = tk.Label(frame, text="Loading...", width=130, height=130, bg="lightgray")
        img_label.grid(row=0, column=0, rowspan=6, padx=10)

        if row.get("image_url"):
            Thread(
                target=load_image_async, args=(img_label, row["image_url"]),
            ).start()

        name_label = tk.Label(
            frame,
            text=row["name"],
            font=("Arial", 12, "bold"),
            wraplength=600,
            justify=tk.LEFT,
        )
        name_label.grid(row=0, column=1, sticky=tk.W, padx=10)

        if row.get("subcategory"):
            subcat_label = tk.Label(
                frame,
                text=row["subcategory"],
                font=("Arial", 9),
                fg="gray",
            )
            subcat_label.grid(row=1, column=1, sticky=tk.W, padx=10)

        price_label = tk.Label(
            frame,
            text=f"Price: ₹{row['price_numeric']:.0f}",
            font=("Arial", 14, "bold"),
            fg="#27ae60",
        )
        price_label.grid(row=2, column=1, sticky=tk.W, padx=10)

        info = f"Source: {row['source']} | Category: {row['category']}"
        if row.get("rating"):
            info += f" | Rating: {row['rating']}"
        if row.get("reviews"):
            info += f" | Reviews: {row['reviews']}"

        info_label = tk.Label(frame, text=info, font=("Arial", 9))
        info_label.grid(row=3, column=1, sticky=tk.W, padx=10)
        
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
        
        sentiment_label = tk.Label(
            frame,
            text=f"{sentiment_emoji} Sentiment: {sentiment.upper()} ({sentiment_score:.0%}) {source_text}",
            font=("Arial", 10, "bold"),
            fg=sentiment_color
        )
        sentiment_label.grid(row=4, column=1, sticky=tk.W, padx=10)
        
        # Sentiment explanation label (why it's positive/negative)
        if sentiment_explanation and sentiment != 'unknown':
            explanation_label = tk.Label(
                frame,
                text=f"🧠 {sentiment_explanation}",
                font=("Arial", 9, "italic"),
                fg="#666"
            )
            explanation_label.grid(row=5, column=1, sticky=tk.W, padx=10)
            next_row = 6
        else:
            next_row = 5
        
        # View Details button
        product_row = row.to_dict()
        
        def show_details(product_data=product_row):
            details_window = tk.Toplevel(root)
            details_window.title(f"Product Details - {product_data['name'][:50]}...")
            details_window.geometry("700x500")
            
            text_widget = scrolledtext.ScrolledText(details_window, wrap=tk.WORD, 
                                                     font=("Consolas", 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            details_text = "="*70 + "\n"
            details_text += f"📱 {product_data['name']}\n"
            details_text += "="*70 + "\n\n"
            
            details_text += f"💰 Price: ₹{product_data['price_numeric']:.0f}\n"
            details_text += f"🏪 Source: {product_data['source']}\n"
            details_text += f"📂 Category: {product_data['category']}\n"
            
            if product_data.get('rating'):
                details_text += f"⭐ Rating: {product_data['rating']}\n"
            if product_data.get('reviews'):
                details_text += f"💬 Reviews: {product_data['reviews']}\n"
            
            # Neural Network Sentiment Analysis info
            sentiment = product_data.get('sentiment', 'unknown')
            sentiment_score = product_data.get('sentiment_score', 0.5)
            sentiment_emoji = product_data.get('sentiment_emoji', '❓')
            sentiment_explanation = product_data.get('sentiment_explanation', '')
            sentiment_source = product_data.get('sentiment_source', 'description')
            
            details_text += f"\n{sentiment_emoji} NEURAL SENTIMENT ANALYSIS (DistilBERT):\n"
            details_text += f"   Sentiment: {sentiment.upper()}\n"
            details_text += f"   Confidence Score: {sentiment_score:.1%}\n"
            details_text += f"   Source: {'Customer Reviews' if sentiment_source == 'customer_reviews' else 'Product Description'}\n"
            
            if product_data.get('sentiment_confidence'):
                conf = product_data['sentiment_confidence']
                if isinstance(conf, dict):
                    details_text += f"   Breakdown: Positive={conf.get('positive', 0):.1%}, "
                    details_text += f"Neutral={conf.get('neutral', 0):.1%}, "
                    details_text += f"Negative={conf.get('negative', 0):.1%}\n"
                elif isinstance(conf, (int, float)):
                    details_text += f"   Confidence: {conf:.1%}\n"
            
            if sentiment_explanation:
                details_text += f"\n   🧠 Analysis: {sentiment_explanation}\n"
            
            details_text += "\n"
            
            # Rating Breakdown (5 star: 61%, etc.)
            rating_breakdown = product_data.get('rating_breakdown')
            if rating_breakdown and isinstance(rating_breakdown, dict):
                details_text += "⭐ RATING BREAKDOWN:\n" + "-"*50 + "\n"
                for star, percent in rating_breakdown.items():
                    details_text += f"  {star}: {percent}\n"
                details_text += "\n"
            
            # Customer Reviews
            if product_data.get('customer_reviews'):
                reviews = product_data['customer_reviews']
                # Ensure reviews is a list
                if isinstance(reviews, list) and len(reviews) > 0:
                    details_text += f"💬 CUSTOMER REVIEWS ({len(reviews)} shown):\n" + "-"*50 + "\n"
                    for idx, review in enumerate(reviews[:5], 1):
                        if isinstance(review, dict):
                            if review.get('rating'):
                                details_text += f"  ⭐ {review['rating']}\n"
                            if review.get('title'):
                                details_text += f"  📌 {review['title']}\n"
                            if review.get('text'):
                                review_text = review['text'][:300] + "..." if len(review['text']) > 300 else review['text']
                                details_text += f"  {review_text}\n"
                        elif isinstance(review, str):
                            details_text += f"  {review[:300]}\n"
                        details_text += "\n"
            
            # Review Summary (Customers say...)
            if product_data.get('review_summary'):
                details_text += "📊 CUSTOMERS SAY:\n" + "-"*50 + "\n"
                details_text += f"  {product_data['review_summary']}\n\n"
            
            # Technical Details
            if product_data.get('technical_details'):
                details_text += "📋 TECHNICAL DETAILS:\n" + "-"*50 + "\n"
                for key, value in product_data['technical_details'].items():
                    details_text += f"  {key}: {value}\n"
                details_text += "\n"
            
            # Features
            if product_data.get('features') and isinstance(product_data['features'], list):
                if len(product_data['features']) > 0:
                    details_text += "✨ FEATURES:\n" + "-"*50 + "\n"
                    for idx, feat in enumerate(product_data['features'], 1):
                        details_text += f"  {idx}. {feat}\n"
                    details_text += "\n"
            
            # Description
            if product_data.get('description'):
                details_text += "📝 DESCRIPTION:\n" + "-"*50 + "\n"
                details_text += product_data['description'] + "\n\n"
            
            # Product Link
            if product_data.get('product_link'):
                details_text += "🔗 PRODUCT LINK:\n" + "-"*50 + "\n"
                details_text += product_data['product_link'] + "\n"
            
            details_text += "\n" + "="*70 + "\n"
            
            text_widget.insert(tk.END, details_text)
            text_widget.config(state=tk.DISABLED)
            
            # Buttons frame
            btn_frame = tk.Frame(details_window)
            btn_frame.pack(pady=5)
            
            if product_data.get('product_link'):
                open_btn = tk.Button(btn_frame, text="🌐 Open in Browser", 
                                    command=lambda: webbrowser.open(product_data['product_link']),
                                    bg="#3498db", fg="white", font=("Arial", 10, "bold"),
                                    padx=15, pady=5)
                open_btn.pack(side=tk.LEFT, padx=5)
            
            def copy_to_clipboard():
                details_window.clipboard_clear()
                details_window.clipboard_append(details_text)
            
            copy_btn = tk.Button(btn_frame, text="📋 Copy Details", 
                                command=copy_to_clipboard, bg="#95a5a6", fg="white",
                                font=("Arial", 10, "bold"), padx=15, pady=5)
            copy_btn.pack(side=tk.LEFT, padx=5)
        
        details_btn = tk.Button(frame, text="📄 View Details", 
                               command=show_details, bg="#3498db", fg="white",
                               font=("Arial", 9, "bold"), cursor="hand2", padx=10, pady=3)
        details_btn.grid(row=5, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Product link - clickable
        if row.get('product_link'):
            link_label = tk.Label(frame, text="🔗 View on Website", 
                                font=("Arial", 9, "underline"), fg="blue",
                                cursor="hand2")
            link_label.grid(row=6, column=1, sticky=tk.W, padx=10, pady=3)
            link_label.bind("<Button-1>", lambda e, url=row['product_link']: webbrowser.open(url))

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    root.mainloop()


# ==========================================================
# AGENTS
# ==========================================================

class BrowserAgent:
    """Manages a single Selenium driver (one per site/thread)."""

    def __init__(self):
        self.driver = None

    def start(self):
        self.driver = setup_driver()
        return self.driver

    def stop(self):
        if self.driver:
            self.driver.quit()
            self.driver = None


class AmazonAgent:
    def __init__(self, browser_agent: BrowserAgent):
        self.browser_agent = browser_agent

    def search(self, product_name: str, max_products: int = 5, fetch_details: bool = True):
        return scrape_amazon_in(self.browser_agent.driver, product_name, max_products, fetch_details)


class FlipkartAgent:
    def __init__(self, browser_agent: BrowserAgent):
        self.browser_agent = browser_agent

    def search(self, product_name: str, max_products: int = 5, fetch_details: bool = True):
        return scrape_flipkart(self.browser_agent.driver, product_name, max_products, fetch_details)


class CromaAgent:
    """Agent for scraping Croma.com"""
    def __init__(self, browser_agent: BrowserAgent):
        self.browser_agent = browser_agent
        self.scraper = CromaScraper(browser_agent.driver)

    def search(self, product_name: str, max_products: int = 5, fetch_details: bool = True):
        try:
            return self.scraper.search_products(product_name, max_products)
        except Exception as e:
            print(f"CromaAgent error: {e}")
            return []


class RelianceAgent:
    """Agent for scraping RelianceDigital.in"""
    def __init__(self, browser_agent: BrowserAgent):
        self.browser_agent = browser_agent
        self.scraper = RelianceDigitalScraper(browser_agent.driver)

    def search(self, product_name: str, max_products: int = 5, fetch_details: bool = True):
        try:
            return self.scraper.search_products(product_name, max_products)
        except Exception as e:
            print(f"RelianceAgent error: {e}")
            return []


class SentimentAgent:
    """Agent for analyzing product sentiment using Neural Network (DistilBERT)"""
    
    def __init__(self):
        self.analyzer = None
        self.is_available = False
        
        if NEURAL_SENTIMENT_AVAILABLE:
            try:
                self.analyzer = NeuralSentimentAnalyzer()
                self.is_available = self.analyzer.is_ready
                if self.is_available:
                    print("✅ Neural Sentiment Analyzer loaded (DistilBERT)")
                else:
                    print("⚠️ Neural model not loaded properly")
            except Exception as e:
                print(f"⚠️ Could not load neural sentiment analyzer: {e}")
    
    def analyze_products(self, products):
        """Add sentiment analysis to list of products using neural network"""
        if not self.is_available or not products or self.analyzer is None:
            # Add default sentiment values
            for product in products:
                product['sentiment'] = 'unknown'
                product['sentiment_score'] = 0.5
                product['sentiment_emoji'] = '❓'
            return products
        
        return self.analyzer.analyze_products_batch(products)


class FilterAgent:
    def __init__(self):
        # Initialize ProductValidator for strict Series/Model verification
        if PRODUCT_VALIDATOR_AVAILABLE:
            self.product_validator = ProductValidator()
        else:
            self.product_validator = None
        
        # for phone queries
        self.phone_related_keywords = {
            "phone", "mobile", "smartphone", "iphone", "galaxy",
            "pixel", "oneplus", "redmi", "realme", "vivo", "oppo", "poco"
        }
        self.generic_terms = {"mobile", "phone", "phones", "smartphone", "smartphones", "cell"}
        self.fallback_include = {"phone", "mobile", "smartphone", "iphone"}
        self.exclude_keywords = [
            "case ",
            "cases ",
            " case",
            "cover ",
            "covers ",
            " cover",
            "back cover",
            "bumper",
            "protective case",
            "protective cover",
            "charger",
            "charging cable",
            "cable for",
            "adapter",
            "tempered glass",
            "tempered ",
            "screen guard",
            "screen protector",
            "glass protector",
            "protector for",
            "pouch",
            "strap",
            "phone stand",
            "mobile stand",
            "skin for",
            "phone holder",
            "car holder",
            "battery pack",
            "lens protector",
            "camera protector",
            "camera lens",
            "powerbank",
            "power bank",
            "data cable",
            "ring holder",
            "popsocket",
            "screen film",
            "keyboard for",
            "car mount",
            "phone mount",
            "charging dock",
        ]

        # for headphone queries
        self.headphone_words = {"headphone", "headphones", "headset", "headsets"}
        self.ear_exclude_for_headphones = {
            "earphone", "earphones",
            "earbud", "earbuds",
            "ear pod", "ear pods", "earpod", "earpods",
            "neckband", "neck band",
            "tws", "true wireless",
            "in-ear", "in ear", "ear stick", "earsticks"
        }

        # generic search stopwords – not used as brand/model tokens
        self.generic_search_words = {
            "with", "for", "and", "or", "the", "a", "an",
            "wireless", "bluetooth", "over", "on", "ear", "ears",
            "mic", "microphone", "bass", "deep", "extra",
            "black", "white", "blue", "red", "green", "grey", "gray",
            "playtime", "hours", "upto", "up", "to", "noise", "cancelling",
            "wired", "without", "type", "c", "charging", "fast",
            "pro", "max", "plus", "ultra", "edition", "series"
        }

    # ---------- QUERY TYPE CHECKS ----------

    def is_phone_query(self, search_term: str) -> bool:
        """Check if the query is about a phone."""
        words = re.findall(r"\b\w+\b", search_term.lower())
        return any(word in self.phone_related_keywords for word in words)

    def is_headphone_query(self, search_term: str) -> bool:
        """Check if the query is about headphones / headsets."""
        words = re.findall(r"\b\w+\b", search_term.lower())
        return any(word in self.headphone_words for word in words)

    # ---------- HELPER: important tokens ----------

    def extract_important_tokens(self, search_term: str):
        tokens = re.findall(r"\b\w+\b", search_term.lower())
        important = [
            t for t in tokens
            if t not in self.generic_search_words
            and t not in self.headphone_words
            and t not in self.generic_terms
            and (len(t) > 2 or t.isdigit())
        ]
        return important

    # ---------- PHONE FILTERING ----------

    def clean_price(self, price_text: str) -> float:
        return clean_price(price_text)

    def filter_only_phones(self, products, search_term):
        if not products:
            return products

        tokens = [
            t
            for t in re.split(r"\W+", search_term.lower())
            if t and t not in self.generic_terms
        ]
        
        # Separate brand tokens from model tokens (numbers)
        brand_tokens = [t for t in tokens if not t.isdigit()]
        
        filtered = []
        for p in products:
            title = p.get("name", "").lower()
            if not title:
                continue

            if any(ex in title for ex in self.exclude_keywords):
                continue

            # For phone queries, require brand match but be flexible on model number
            if brand_tokens:
                # At least one brand token must match
                if any(tok in title for tok in brand_tokens):
                    filtered.append(p)
            elif tokens:
                # If only numbers in search, use original logic
                if any(tok in title for tok in tokens):
                    filtered.append(p)
            else:
                if any(f in title for f in self.fallback_include):
                    filtered.append(p)

        return filtered

    # ---------- HEADPHONE FILTERING ----------

    def filter_only_headphones(self, products):
        """Keep only true headphones / headsets."""
        if not products:
            return products

        filtered = []
        for p in products:
            title = (p.get("name", "") + " " + p.get("subcategory", "")).lower()
            if not title:
                continue

            if not any(w in title for w in self.headphone_words):
                continue

            if any(ex in title for ex in self.ear_exclude_for_headphones):
                continue

            filtered.append(p)

        return filtered

    # ---------- SEARCH-TERM RELEVANCE FILTER ----------

    def verify_by_search_term(self, products, search_term):
        """
        Second-stage verification:
        - Extract brand/model tokens from search term
        - For phone queries, be more flexible (match brand, not model number)
        - If such tokens exist, keep only products whose title contains
          the important non-numeric tokens (removes unrelated 'advertised' items).
        - If no important tokens (e.g. search is just 'headphones'), return as-is.
        """
        if not products:
            return products

        important = self.extract_important_tokens(search_term)
        if not important:
            return products

        # Separate brand/text tokens from numeric tokens (model numbers)
        brand_tokens = [t for t in important if not t.isdigit()]
        
        # If no brand tokens, return as-is (search was just numbers)
        if not brand_tokens:
            return products

        filtered = []
        for p in products:
            text = (p.get("name", "") + " " + p.get("subcategory", "") + " " + p.get("description", "")).lower()
            # Require at least one brand token to match (more lenient to avoid losing products)
            # Also check source to keep all Amazon/Flipkart/Croma/Reliance products
            source = p.get("source", "").lower()
            if any(tok in text for tok in brand_tokens):
                filtered.append(p)
            # Keep products with valid sources even if name is truncated
            elif source in ["amazon.in", "flipkart", "croma", "reliance digital"]:
                # Secondary check: ensure product is somewhat relevant
                if any(tok in text for tok in brand_tokens) or len(text) > 10:
                    filtered.append(p)

        return filtered if filtered else products

    # ---------- DATAFRAME BUILD ----------

    def build_dataframe(self, products):
        if not products:
            return pd.DataFrame()

        df = pd.DataFrame(products)
        if "price" not in df.columns:
            return pd.DataFrame()

        df["price_numeric"] = df["price"].apply(self.clean_price)
        df = df[df["price_numeric"] > 0]
        if df.empty:
            return df

        return df.sort_values(by="price_numeric")


class GUIAgent:
    @staticmethod
    def show(df: pd.DataFrame):
        display_results_gui(df)


# ==========================================================
# COORDINATOR – MULTI-AGENT WORKFLOW
# ==========================================================

def multi_agent_compare_prices(product_name: str, max_products: int = 5, fetch_details: bool = True):
    """
    Multi-agent price comparison with parallel scraping.
    
    Args:
        product_name: Product to search for
        max_products: Maximum products per site
        fetch_details: If True, scrape detailed product info (slower). 
                      If False, only scrape search page (faster).
    """
    # Start power monitoring
    power_monitor = None
    if POWER_MONITOR_AVAILABLE:
        power_monitor = PowerMonitor()
        power_monitor.start_monitoring()
    # Create browser agents for all platforms
    amazon_browser = BrowserAgent()
    flipkart_browser = BrowserAgent()
    croma_browser = BrowserAgent()
    reliance_browser = BrowserAgent()

    amazon_browser.start()
    flipkart_browser.start()
    croma_browser.start()
    reliance_browser.start()

    amazon_agent = AmazonAgent(amazon_browser)
    flipkart_agent = FlipkartAgent(flipkart_browser)
    croma_agent = CromaAgent(croma_browser)
    reliance_agent = RelianceAgent(reliance_browser)
    filter_agent = FilterAgent()
    sentiment_agent = SentimentAgent()  # Neural Network sentiment analyzer
    gui_agent = GUIAgent()

    amazon_products = []
    flipkart_products = []
    croma_products = []
    reliance_products = []

    def run_amazon():
        nonlocal amazon_products
        amazon_products = amazon_agent.search(product_name, max_products, fetch_details)

    def run_flipkart():
        nonlocal flipkart_products
        flipkart_products = flipkart_agent.search(product_name, max_products, fetch_details)

    def run_croma():
        nonlocal croma_products
        croma_products = croma_agent.search(product_name, max_products, fetch_details)

    def run_reliance():
        nonlocal reliance_products
        reliance_products = reliance_agent.search(product_name, max_products, fetch_details)

    # Run all scrapers in parallel
    t1 = Thread(target=run_amazon)
    t2 = Thread(target=run_flipkart)
    t3 = Thread(target=run_croma)
    t4 = Thread(target=run_reliance)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    amazon_browser.stop()
    flipkart_browser.stop()
    croma_browser.stop()
    reliance_browser.stop()

    print(f"Amazon products scraped: {len(amazon_products)}")
    print(f"Flipkart products scraped: {len(flipkart_products)}")
    print(f"Croma products scraped: {len(croma_products)}")
    print(f"Reliance Digital products scraped: {len(reliance_products)}")

    all_products = amazon_products + flipkart_products + croma_products + reliance_products
    if not all_products:
        print("No products scraped from either site.")
        return

    # === STRICT PRODUCT VALIDATION ===
    # Brand + Product Line = OK (no strict verification)
    # Series/Model Number = MUST be strictly verified
    # Example: Searching "Vivo V30" rejects "Vivo Y100" (V≠Y) and "Vivo V40" (30≠40)
    if filter_agent.product_validator:
        print(f"\n🔍 Validating products with strict Series/Model verification...")
        before_count = len(all_products)
        all_products, rejected = filter_agent.product_validator.filter_products(product_name, all_products)
        removed = len(rejected)
        if removed > 0:
            print(f"✓ ProductValidator: Removed {removed} products with wrong Series/Model")
    
    # Type-based filters
    if filter_agent.is_phone_query(product_name):
        all_products = filter_agent.filter_only_phones(all_products, product_name)
        if not all_products:
            print("No phone-like products found after filtering.")
            return
    elif filter_agent.is_headphone_query(product_name):
        all_products = filter_agent.filter_only_headphones(all_products)
        if not all_products:
            print("No headphone-only products found after filtering.")
            return

    # Verification filter against search term (removes off-brand ads)
    all_products = filter_agent.verify_by_search_term(all_products, product_name)
    
    # Analyze sentiment for all products using neural network
    all_products = sentiment_agent.analyze_products(all_products)

    df = filter_agent.build_dataframe(all_products)
    if df.empty:
        print("No products with valid prices found.")
        return

    print(f"\nFound {len(df)} products after cleaning")
    print(
        f"Price range: ₹{df['price_numeric'].min():.0f} - ₹{df['price_numeric'].max():.0f}"
    )
    
    # === UMAP Clustering Visualization ===
    if UMAP_ANALYZER_AVAILABLE and len(all_products) >= 10:
        print(f"\n{'='*60}")
        print("🗺️  UMAP Clustering Visualization...")
        print("-"*60)
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
    
    # === Power Consumption Report ===
    if power_monitor:
        print(f"\n{'='*60}")
        print("⚡ Power Consumption Report...")
        print("-"*60)
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
    
    print("\nOpening GUI...")
    gui_agent.show(df)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    product_name = input("Enter product name to search: ")
    max_products_input = input("How many products per source? (default: 5): ").strip()
    max_products = int(max_products_input) if max_products_input.isdigit() else 5
    
    fetch_details_input = input("Fetch detailed product info? (y/n, default: y): ").strip().lower()
    fetch_details = fetch_details_input != 'n'
    
    if not fetch_details:
        print("⚡ Fast mode: Skipping detailed product page scraping for faster results")
    else:
        print("📋 Full mode: Will scrape detailed specs from product pages (slower)")
    
    multi_agent_compare_prices(product_name, max_products, fetch_details)
