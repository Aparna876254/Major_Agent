from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
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

# ===========================
# Enhanced Web Scraping
# ===========================

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

def setup_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": random.choice(user_agents)})
    except Exception:
        pass
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except Exception:
        pass
    driver.set_page_load_timeout(30)

    return driver


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
            'features': []
        }
        
        print(f"      Searching for Amazon product details...")
        
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
    """Scrape detailed information from Flipkart product page"""
    try:
        # Only navigate if URL is provided (for backward compatibility)
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': []
        }
        
        print(f"      Searching for Flipkart product details...")
        
        # Keywords to search for in Flipkart pages
        spec_keywords = ['specification', 'specifications', 'specs']
        desc_keywords = ['description', 'product description', 'about']
        detail_keywords = ['product detail', 'product details', 'details', 'product information']
        
        # Method 1: Try to find elements by searching text content
        try:
            # Find all headings and check for keywords
            all_headings = driver.find_elements(By.CSS_SELECTOR, "div, span, h1, h2, h3")
            spec_sections = []
            
            for heading in all_headings:
                try:
                    text = heading.text.strip().lower()
                    
                    # Check if heading contains specification keywords
                    if any(keyword in text for keyword in spec_keywords + detail_keywords):
                        # Found a specification section
                        parent = heading
                        for _ in range(3):  # Go up 3 levels to find the container
                            try:
                                parent = parent.find_element(By.XPATH, "..")
                                # Check if this parent has table rows
                                rows = parent.find_elements(By.CSS_SELECTOR, "tr, li")
                                if rows:
                                    spec_sections.append(parent)
                                    print(f"      Found section by keyword '{text[:30]}': {len(rows)} rows")
                                    break
                            except:
                                break
                except:
                    continue
            
            # Extract from found sections
            for section in spec_sections:
                try:
                    # Try to get section title
                    section_title = ""
                    try:
                        title_elem = section.find_element(By.CSS_SELECTOR, "div._4BJ2V\\+, div._1AtVbE, div._3dtsli, span, h3")
                        section_title = title_elem.text.strip()
                    except:
                        section_title = "Specifications"
                    
                    # Get rows
                    rows = section.find_elements(By.CSS_SELECTOR, "tr")
                    for row in rows:
                        try:
                            cells = row.find_elements(By.CSS_SELECTOR, "td")
                            if len(cells) >= 2:
                                key = cells[0].text.strip()
                                value = cells[1].text.strip()
                                if key and value and len(key) < 100:
                                    details['technical_details'][f"{section_title} - {key}"] = value
                        except:
                            continue
                except:
                    continue
                    
        except Exception as e:
            print(f"      Error in keyword search: {e}")
        
        # Method 2: Try multiple CSS selectors for specifications sections
        if not details['technical_details']:
            spec_section_selectors = [
                "div._9V0sS6",  # New Flipkart layout
                "div.GNDEQ-",
                "div._1s76Cw",
                "div._3dtsli",
                "div.aMraIH",
                "div[class*='spec']",  # Any class containing 'spec'
                "div[class*='detail']"  # Any class containing 'detail'
            ]
            
            spec_sections = []
            for selector in spec_section_selectors:
                try:
                    sections = driver.find_elements(By.CSS_SELECTOR, selector)
                    if sections:
                        spec_sections = sections
                        print(f"      Found {len(sections)} spec sections using selector: {selector}")
                        break
                except:
                    continue
            
            # Extract specifications from found sections
            if spec_sections:
                for section_idx, section in enumerate(spec_sections):
                    try:
                        # Try to find section title
                        section_title = ""
                        title_selectors = [
                            "div._4BJ2V\\+",  # New layout
                            "div._1AtVbE",
                            "div._3dtsli", 
                            "div._2RngUh",
                            "span"
                        ]
                        
                        for title_sel in title_selectors:
                            try:
                                title_elem = section.find_element(By.CSS_SELECTOR, title_sel)
                                section_title = title_elem.text.strip()
                                if section_title and len(section_title) > 0 and len(section_title) < 50:
                                    break
                            except:
                                continue
                        
                        if not section_title:
                            section_title = f"Section {section_idx + 1}"
                        
                        print(f"      Processing section: {section_title}")
                        
                        # Try to find specification rows
                        row_selectors = [
                            "tr.WJdYP6",  # New layout
                            "tr._2-N8s",
                            "li.W5FkOm",
                            "div.row",
                            "tr"  # Generic tr
                        ]
                        
                        rows = []
                        for row_sel in row_selectors:
                            try:
                                found_rows = section.find_elements(By.CSS_SELECTOR, row_sel)
                                if found_rows:
                                    rows = found_rows
                                    print(f"        Found {len(rows)} rows using selector: {row_sel}")
                                    break
                            except:
                                continue
                        
                        # Extract key-value pairs from rows
                        for row in rows:
                            try:
                                # Try multiple selectors for table cells
                                cell_selectors = [
                                    "td.URwL2w",  # New layout - key
                                    "td._7eSDEY",  # New layout - value
                                    "td._2H-kL",
                                    "td._2vIOIi",
                                    "td",  # Generic td
                                    "li._21Ahn-"
                                ]
                                
                                # Try to get key and value
                                cells = []
                                for cell_sel in cell_selectors:
                                    try:
                                        found_cells = row.find_elements(By.CSS_SELECTOR, cell_sel)
                                        if found_cells:
                                            cells.extend(found_cells)
                                            break  # Stop after first successful selector
                                    except:
                                        continue
                                
                                # If we have at least 2 cells, treat as key-value
                                if len(cells) >= 2:
                                    key = cells[0].text.strip()
                                    value = cells[1].text.strip()
                                    if key and value and len(key) < 100:
                                        details['technical_details'][f"{section_title} - {key}"] = value
                                        print(f"          {key}: {value[:50]}")
                                elif len(cells) == 1:
                                    # Single cell - might be a feature/highlight
                                    text = cells[0].text.strip()
                                    if text and len(text) > 3:
                                        details['features'].append(text)
                            except Exception as row_error:
                                continue
                    
                    except Exception as section_error:
                        print(f"      Error in section: {section_error}")
                        continue
        
        # Description - search by keywords
        desc_selectors = [
            "div._1mXcCf",
            "div._2418kt",
            "div.yN\\+eNk",
            "div._2RngUh p",
            "div.product-description",
            "div[class*='description']",  # Any class containing 'description'
            "div[class*='desc']"  # Any class containing 'desc'
        ]
        
        for desc_sel in desc_selectors:
            try:
                desc_element = driver.find_element(By.CSS_SELECTOR, desc_sel)
                details['description'] = desc_element.text.strip()
                if details['description']:
                    print(f"      Found description using: {desc_sel}")
                    break
            except:
                continue
        
        # If no description found, search by text
        if not details['description']:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    try:
                        div_class = div.get_attribute("class") or ""
                        if any(kw in div_class.lower() for kw in desc_keywords):
                            text = div.text.strip()
                            if text and len(text) > 50 and len(text) < 2000:
                                details['description'] = text
                                print(f"      Found description by keyword in: {div_class[:30]}")
                                break
                    except:
                        continue
            except:
                pass
        
        # Highlights/Features - try multiple selectors
        highlight_selectors = [
            "li._21Ahn-",
            "li.WJdYP6",
            "ul._1D2qrc li",
            "div._2418kt ul li",
            "li[class*='highlight']",  # Any class containing 'highlight'
            "li[class*='feature']"  # Any class containing 'feature'
        ]
        
        for hl_sel in highlight_selectors:
            try:
                highlights = driver.find_elements(By.CSS_SELECTOR, hl_sel)
                if highlights:
                    for hl in highlights:
                        text = hl.text.strip()
                        if text and len(text) > 3 and text not in details['features']:
                            details['features'].append(text)
                    if details['features']:
                        print(f"      Found {len(details['features'])} features using: {hl_sel}")
                        break
            except:
                continue
        
        # Combine features into description if description is empty
        if not details['description'] and details['features']:
            details['description'] = ' | '.join(details['features'])
        
        # Print summary
        print(f"      Flipkart extraction complete: {len(details['technical_details'])} specs, {len(details['features'])} features")
        
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
            search_tokens_lower = set(product_name.lower().split()) - {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy', 'watch', 'smart', 'smartwatch'}
            if search_tokens_lower and len(search_tokens_lower) > 0:
                brands = ['samsung', 'apple', 'iphone', 'oneplus', 'xiaomi', 'redmi', 'realme', 'oppo', 'vivo', 'pixel', 'galaxy', 'noise', 'titan', 'boat']
                has_brand = any(brand in product_name.lower() for brand in brands)
                if has_brand:
                    brand_match = any(term in name_lower for term in search_tokens_lower)
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
                        # Try to get name from link title attribute first
                        name = link_elem.get_attribute("title") or link_elem.get_attribute("aria-label") or link_elem.text.strip()
                        if product_link:
                            print(f"    Found link: {product_link[:60]}...")
                        if name and len(name) > 5:
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
                # Try to get all text from the item and extract product name
                try:
                    item_text = item.text
                    lines = [line.strip() for line in item_text.split('\n') if line.strip()]
                    for line in lines:
                        # Skip lines that are clearly not product names
                        if (len(line) > 15 and 
                            not line.startswith('₹') and 
                            not any(gen in line.lower() for gen in generic_names) and
                            not line.replace(',', '').replace('(', '').replace(')', '').isdigit() and
                            'rating' not in line.lower() and
                            '%' not in line):
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
                                print(f"    Found name with selector '{sel}': {name[:60]}")
                                break
                        except:
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
            search_tokens_lower = set(product_name.lower().split()) - {'mobile', 'phone', 'smartphone', 'the', 'a', 'an', 'best', 'new', 'latest', 'buy', 'watch', 'smart', 'smartwatch'}
            if search_tokens_lower and len(search_tokens_lower) > 0:
                brands = ['samsung', 'apple', 'iphone', 'oneplus', 'xiaomi', 'redmi', 'realme', 'oppo', 'vivo', 'pixel', 'galaxy', 'noise', 'titan', 'boat']
                has_brand = any(brand in product_name.lower() for brand in brands)
                if has_brand:
                    brand_match = any(term in name_lower for term in search_tokens_lower)
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
        
        # Description/Features
        if row.get('description'):
            desc_text = row['description'][:200] + "..." if len(row.get('description', '')) > 200 else row.get('description', '')
            desc_label = tk.Label(product_frame, text=f"Description: {desc_text}", 
                                font=("Arial", 9), wraplength=800, justify=tk.LEFT,
                                bg=product_frame['bg'], fg="#333")
            desc_label.grid(row=3, column=1, sticky=tk.W, padx=10, pady=3)
        
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
                
                details_text += "\n" + "="*90 + "\n\n"
                
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
            details_btn.grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Product link
        if row.get('product_link'):
            def open_link(url=row['product_link']):
                webbrowser.open(url)
            
            link_label = tk.Label(product_frame, text="🔗 View on Website", 
                                font=("Arial", 9, "underline"), fg="blue",
                                bg=product_frame['bg'], cursor="hand2")
            link_label.grid(row=5, column=1, sticky=tk.W, padx=10, pady=3)
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
    
    driver = setup_driver()
    
    try:
        print("Scraping Amazon with Full Product Details...")
        print("-"*70)
        amazon_products = scrape_detailed_amazon(driver, product_name, max_products)
        
        print("\nScraping Flipkart with Full Product Details...")
        print("-"*70)
        flipkart_products = scrape_detailed_flipkart(driver, product_name, max_products)
        
    finally:
        driver.quit()

    all_products = amazon_products + flipkart_products
    
    # Filter 1: Validate products match search query
    print(f"\n🔍 Validating scraped products match search query...")
    validated_products = []
    for p in all_products:
        p_name = str(p.get('name', '')).lower()
        if search_tokens:
            match_count = sum(1 for term in search_tokens if term in p_name)
            if match_count >= len(search_tokens) * 0.6:
                validated_products.append(p)
        else:
            validated_products.append(p)
    
    removed = len(all_products) - len(validated_products)
    if removed > 0:
        print(f"✓ Removed {removed} irrelevant products, kept {len(validated_products)} matching")
    
    # If too many removed, scrape more to reach target
    target_count = max_products * 2
    if removed > 0 and len(validated_products) < target_count:
        shortage = target_count - len(validated_products)
        print(f"⚠️  Need {shortage} more products. Re-scraping...")
        
        driver = setup_driver()
        try:
            extra_amazon = scrape_detailed_amazon(driver, product_name, shortage)
            extra_flipkart = scrape_detailed_flipkart(driver, product_name, shortage)
        finally:
            driver.quit()
        
        extra_products = extra_amazon + extra_flipkart
        for p in extra_products:
            p_name = str(p.get('name', '')).lower()
            if search_tokens:
                match_count = sum(1 for term in search_tokens if term in p_name)
                if match_count >= len(search_tokens) * 0.6:
                    validated_products.append(p)
        
        print(f"✓ Added {len(validated_products) - (len(all_products) - removed)} more products")
    
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
    print("\n📋 Main Menu:")
    print("1. 🔍 Search Products (Unified RAG: Local → Fuzzy → Web Scraping)")
    print("2. 📊 View Database Statistics")
    print("3. 🗑️  Clear Database")
    print("4. 🚪 Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
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
            # Exit
            print("\n👋 Exiting... Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

