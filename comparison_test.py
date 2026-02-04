"""
Comprehensive Comparison Test: Single Agent vs Multi-Agent E-Commerce Scraper

This script performs a complete comparison test for the Major Project Results section.
It collects detailed metrics on time, bandwidth, accuracy, and energy efficiency.

Usage:
    python comparison_test.py

Output:
    - scraped_single_agent.pkl
    - scraped_multi_agent.pkl
"""

import pickle
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from threading import Thread
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ==========================================================
# TEST CONFIGURATIONS
# ==========================================================

# Test Configuration - 10 Categories with 4 queries each (Standard)
CATEGORIES_STANDARD = [
    ("Smartphones", ["iPhone 15", "Samsung Galaxy S24", "OnePlus 12", "Pixel 8"]),
    ("Laptops", ["Dell Inspiron", "HP Pavilion", "Lenovo ThinkPad", "MacBook Air"]),
    ("Smartwatches", ["Apple Watch Series 9", "Samsung Galaxy Watch", "Noise ColorFit", "Boat Storm"]),
    ("Tablets", ["iPad", "Samsung Galaxy Tab", "Lenovo Tab", "Redmi Pad"]),
    ("Wireless Earbuds", ["AirPods Pro", "Samsung Galaxy Buds", "Boat Airdopes", "Noise Buds"]),
    ("Headphones", ["Sony WH-1000XM5", "Bose QuietComfort", "JBL Tune", "Sennheiser"]),
    ("Smart TVs", ["Samsung Smart TV 55", "LG OLED TV", "Sony Bravia", "Mi TV"]),
    ("Cameras", ["Canon EOS", "Nikon D", "Sony Alpha", "GoPro Hero"]),
    ("Gaming Consoles", ["PlayStation 5", "Xbox Series X", "Nintendo Switch", "Gaming Laptop"]),
    ("Smart Speakers", ["Amazon Echo", "Google Nest", "Apple HomePod", "JBL Flip"]),
]

# Large Categories Test - Extended queries for popular categories (Smartphones, Laptops, etc.)
CATEGORIES_LARGE = [
    ("Smartphones", [
        "iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15", "iPhone 14",
        "Samsung Galaxy S24 Ultra", "Samsung Galaxy S24+", "Samsung Galaxy S24", "Samsung Galaxy A54",
        "OnePlus 12", "OnePlus 12R", "OnePlus Nord 4", "OnePlus Nord CE 4",
        "Google Pixel 8 Pro", "Google Pixel 8", "Google Pixel 7a",
        "Xiaomi 14", "Redmi Note 13 Pro+", "Redmi Note 13", "Poco X6 Pro",
        "Vivo V30 Pro", "Vivo V30", "Oppo Reno 11 Pro", "Realme GT 5 Pro", "Nothing Phone 2"
    ]),
    ("Laptops", [
        "MacBook Air M3", "MacBook Pro 14 M3", "MacBook Pro 16 M3",
        "Dell XPS 15", "Dell XPS 13", "Dell Inspiron 15", "Dell Latitude 5540",
        "HP Spectre x360", "HP Pavilion 15", "HP Envy x360", "HP Victus Gaming",
        "Lenovo ThinkPad X1 Carbon", "Lenovo ThinkPad E14", "Lenovo IdeaPad Slim 5", "Lenovo Legion 5",
        "ASUS ROG Zephyrus G14", "ASUS VivoBook 15", "ASUS TUF Gaming F15",
        "Acer Nitro 5", "Acer Aspire 7", "MSI Gaming Laptop", "Microsoft Surface Laptop 5"
    ]),
    ("Tablets", [
        "iPad Pro 12.9", "iPad Pro 11", "iPad Air M2", "iPad 10th Gen", "iPad Mini 6",
        "Samsung Galaxy Tab S9 Ultra", "Samsung Galaxy Tab S9+", "Samsung Galaxy Tab S9", "Samsung Galaxy Tab A9",
        "OnePlus Pad", "Xiaomi Pad 6", "Redmi Pad SE", "Lenovo Tab P12 Pro",
        "Amazon Fire HD 10", "Microsoft Surface Pro 9"
    ]),
    ("Smartwatches", [
        "Apple Watch Ultra 2", "Apple Watch Series 9", "Apple Watch SE 2",
        "Samsung Galaxy Watch 6 Classic", "Samsung Galaxy Watch 6", "Samsung Galaxy Watch 5",
        "Google Pixel Watch 2", "Garmin Venu 3", "Garmin Fenix 7",
        "Amazfit GTR 4", "Amazfit Balance", "Noise ColorFit Pro 5", "Boat Wave Prime"
    ]),
    ("Wireless Earbuds", [
        "Apple AirPods Pro 2", "Apple AirPods 3", "Samsung Galaxy Buds2 Pro", "Samsung Galaxy Buds FE",
        "Sony WF-1000XM5", "Sony WF-C700N", "Bose QuietComfort Ultra Earbuds",
        "Google Pixel Buds Pro", "OnePlus Buds 3", "Nothing Ear 2",
        "Jabra Elite 85t", "Sennheiser Momentum True Wireless 4",
        "Boat Airdopes 141", "Boat Airdopes 311", "Noise Buds VS104"
    ])
]

# Quick Test - Minimal queries for rapid testing
CATEGORIES_QUICK = [
    ("Smartphones", ["iPhone 15", "Samsung Galaxy S24"]),
    ("Laptops", ["MacBook Air", "Dell XPS 15"]),
]

# Default configuration
CATEGORIES = CATEGORIES_STANDARD

PRODUCTS_PER_QUERY = 1
TOTAL_QUERIES = len(CATEGORIES) * 4

# India electricity carbon factor
CO2_FACTOR_INDIA = 0.820  # kg CO2/kWh


def setup_driver():
    """Setup Chrome WebDriver with optimized settings (like Try.py)"""
    chrome_options = Options()
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Headless mode for comparison test
    chrome_options.add_argument('--headless=new')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        })
        driver.implicitly_wait(10)
        return driver
    except Exception as e:
        print(f"Driver setup error: {e}")
        return None

# Import analyzers
try:
    from umap_rag_analyzer import UMAPAnalyzer, UMAP_AVAILABLE
    UMAP_ANALYZER_AVAILABLE = UMAP_AVAILABLE
except ImportError:
    UMAP_ANALYZER_AVAILABLE = False
    print("⚠️ UMAP analyzer not available")

try:
    from power_monitor import PowerMonitor
    POWER_MONITOR_AVAILABLE = True
except ImportError:
    POWER_MONITOR_AVAILABLE = False
    print("⚠️ Power monitor not available")

# Import sentiment analyzer for multi-agent post-processing
try:
    from neural_sentiment_analyzer import NeuralSentimentAnalyzer, TRANSFORMERS_AVAILABLE
    NEURAL_SENTIMENT_AVAILABLE = TRANSFORMERS_AVAILABLE
except ImportError:
    NEURAL_SENTIMENT_AVAILABLE = False
    print("⚠️ Neural sentiment analyzer not available")


def add_sentiment_to_products(products: List[Dict]) -> List[Dict]:
    """Add sentiment analysis to products that don't have it"""
    if not NEURAL_SENTIMENT_AVAILABLE:
        print("⚠️ Sentiment analysis skipped (transformers not available)")
        return products
    
    print("\n🧠 Adding sentiment analysis to products...")
    try:
        analyzer = NeuralSentimentAnalyzer()
        
        for i, product in enumerate(products):
            # Skip if already has sentiment
            if product.get('sentiment_score'):
                continue
            
            # Combine text for sentiment analysis
            text_parts = []
            if product.get('name'):
                text_parts.append(str(product['name']))
            if product.get('description'):
                text_parts.append(str(product['description'])[:500])
            if product.get('customer_reviews'):
                reviews = product['customer_reviews']
                if isinstance(reviews, list):
                    for r in reviews[:3]:
                        if isinstance(r, dict):
                            text_parts.append(str(r.get('text', ''))[:200])
                        else:
                            text_parts.append(str(r)[:200])
                else:
                    text_parts.append(str(reviews)[:500])
            
            text = ' '.join(text_parts)[:1000]
            
            if text.strip():
                result = analyzer.analyze_sentiment(text)
                product['sentiment'] = result.get('sentiment', 'neutral')
                product['sentiment_score'] = result.get('score', 0.5)
                product['sentiment_label'] = result.get('sentiment', 'neutral').capitalize()
                product['sentiment_confidence'] = result.get('confidence', 0.5) * 100
            else:
                product['sentiment'] = 'neutral'
                product['sentiment_score'] = 0.5
                product['sentiment_label'] = 'Neutral'
                product['sentiment_confidence'] = 50.0
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i+1}/{len(products)} products...")
        
        print(f"✅ Sentiment analysis complete for {len(products)} products")
        
    except Exception as e:
        print(f"⚠️ Sentiment analysis error: {e}")
    
    return products


def add_price_numeric(products: List[Dict]) -> List[Dict]:
    """Add price_numeric field to products"""
    import re
    
    for product in products:
        if not product.get('price_numeric'):
            price_str = str(product.get('price', ''))
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹$,\s]', '', price_str)
            match = re.search(r'[\d.]+', cleaned)
            if match:
                try:
                    product['price_numeric'] = float(match.group())
                except ValueError:
                    product['price_numeric'] = None
    
    return products


class ComprehensiveMetrics:
    """Collect comprehensive metrics for comparison"""
    
    def __init__(self, method_name: str):
        self.method = method_name
        self.start_time = None
        self.end_time = None
        self.queries = []
        self.products_per_source = {'Amazon': 0, 'Flipkart': 0, 'Croma': 0, 'Reliance Digital': 0}
        self.products_per_category = {}
        self.errors = 0
        self.error_details = []
        self.power_data = {}
        self.umap_data = {}
        
    def start(self):
        self.start_time = datetime.now()
        
    def end(self):
        self.end_time = datetime.now()
        
    def add_query_result(self, query: str, category: str, products: List[Dict], 
                         query_time: float, sources: Dict[str, int]):
        self.queries.append({
            'query': query,
            'category': category,
            'products': len(products),
            'time': query_time,
            'sources': sources
        })
        
        if category not in self.products_per_category:
            self.products_per_category[category] = 0
        self.products_per_category[category] += len(products)
        
        for source, count in sources.items():
            if source in self.products_per_source:
                self.products_per_source[source] += count
                
    def add_error(self, query: str, error: str):
        self.errors += 1
        self.error_details.append({'query': query, 'error': str(error)})
        
    def get_summary(self) -> Dict:
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        total_products = sum(self.products_per_source.values())
        query_times = [q['time'] for q in self.queries if q['time'] > 0]
        
        return {
            'method': self.method,
            'total_products': total_products,
            'total_time': total_time,
            'total_time_minutes': total_time / 60,
            'total_queries': len(self.queries),
            'avg_time_per_query': np.mean(query_times) if query_times else 0,
            'avg_products_per_query': total_products / max(1, len(self.queries)),
            'throughput_products_per_second': total_products / max(1, total_time),
            'avg_time_per_product': total_time / max(1, total_products),
            'products_per_source': self.products_per_source,
            'products_per_category': self.products_per_category,
            'errors': self.errors,
            'success_rate': (1 - self.errors / max(1, len(self.queries))) * 100,
            'queries': self.queries,
            'power': self.power_data,
            'umap': self.umap_data,
            'latency_mean': np.mean(query_times) if query_times else 0,
            'latency_median': np.median(query_times) if query_times else 0,
            'latency_min': np.min(query_times) if query_times else 0,
            'latency_max': np.max(query_times) if query_times else 0,
            'latency_std': np.std(query_times) if query_times else 0,
            'latency_p95': np.percentile(query_times, 95) if query_times else 0,
            'latency_p99': np.percentile(query_times, 99) if query_times else 0,
            'latency_variance': np.var(query_times) if query_times else 0,
        }


def run_single_agent_test() -> Tuple[List[Dict], Dict]:
    """Run comprehensive single-agent test using Try.py"""
    print("\n" + "=" * 70)
    print("   TEST 1: SINGLE AGENT (Try.py)")
    print("=" * 70)
    
    metrics = ComprehensiveMetrics("Single Agent (Try.py)")
    
    power_monitor = None
    if POWER_MONITOR_AVAILABLE:
        power_monitor = PowerMonitor()
        power_monitor.start_monitoring()
    
    try:
        from Single_Agent_Scrapper import unified_rag_search, ProductRAGStorage
    except ImportError as e:
        print(f"❌ Could not import Try.py: {e}")
        return [], {}
    
    test_storage = ProductRAGStorage('test_single_agent_rag.pkl')
    all_products = []
    
    metrics.start()
    query_count = 0
    
    for category, queries in CATEGORIES:
        print(f"\n📂 Category: {category}")
        
        for query in queries:
            query_count += 1
            print(f"\n   [{query_count}/{TOTAL_QUERIES}] Searching: {query}")
            
            query_start = time.time()
            sources = {'Amazon': 0, 'Flipkart': 0, 'Croma': 0, 'Reliance Digital': 0}
            
            try:
                df = unified_rag_search(query, test_storage, max_products=PRODUCTS_PER_QUERY)
                query_time = time.time() - query_start
                
                if df is not None and not df.empty:
                    products = df.to_dict('records')
                    for p in products:
                        p['category'] = category
                        p['search_query'] = query
                        p['scrape_time'] = query_time
                        
                        source = p.get('source', 'Unknown')
                        if 'Amazon' in source:
                            sources['Amazon'] += 1
                        elif 'Flipkart' in source:
                            sources['Flipkart'] += 1
                        elif 'Croma' in source:
                            sources['Croma'] += 1
                        elif 'Reliance' in source:
                            sources['Reliance Digital'] += 1
                    
                    all_products.extend(products)
                    metrics.add_query_result(query, category, products, query_time, sources)
                    print(f"      ✓ Found {len(products)} products in {query_time:.1f}s")
                else:
                    metrics.add_query_result(query, category, [], query_time, sources)
                    print(f"      ⚠️ No products found")
                    
            except Exception as e:
                query_time = time.time() - query_start
                metrics.add_error(query, str(e))
                metrics.add_query_result(query, category, [], query_time, sources)
                print(f"      ❌ Error: {e}")
    
    metrics.end()
    
    if power_monitor:
        try:
            power_monitor.record_measurement("Single Agent Complete")
            power_report = power_monitor.generate_report()
            metrics.power_data = {
                'avg_cpu_percent': power_report.get('resource_utilization', {}).get('average_cpu_usage_percent', 0),
                'avg_cpu_power_watts': power_report.get('power_consumption', {}).get('average_cpu_power_watts', 0),
                'avg_memory_percent': power_report.get('resource_utilization', {}).get('average_memory_usage_percent', 0),
                'total_energy_kwh': power_report.get('energy_consumption', {}).get('total_energy_kwh', 0),
                'co2_emissions_grams': power_report.get('co2_emissions_grams', {}).get('india', 0),
            }
            print(f"\n⚡ Power: {metrics.power_data['avg_cpu_power_watts']:.2f}W, {metrics.power_data['total_energy_kwh']:.6f} kWh")
        except Exception as e:
            print(f"⚠️ Power monitoring error: {e}")
    
    if UMAP_ANALYZER_AVAILABLE and len(all_products) >= 10:
        try:
            print("\n🗺️ Running UMAP clustering...")
            analyzer = UMAPAnalyzer(all_products)
            analyzer.prepare_features()
            analyzer.run_umap()
            umap_metrics = analyzer.calculate_clustering_metrics()
            metrics.umap_data = {
                'silhouette_score': umap_metrics.get('silhouette_score', 0),
                'davies_bouldin_index': umap_metrics.get('davies_bouldin_index', 0),
                'cluster_purity': umap_metrics.get('cluster_purity', 0),
                'n_clusters': umap_metrics.get('n_clusters', 0)
            }
            analyzer.create_visualization('umap_single_agent.png')
            print(f"✓ UMAP saved: umap_single_agent.png")
        except Exception as e:
            print(f"⚠️ UMAP error: {e}")
    
    summary = metrics.get_summary()
    with open('scraped_single_agent.pkl', 'wb') as f:
        pickle.dump({
            'products': all_products,
            'metrics': summary,
            'scraped_at': datetime.now().isoformat()
        }, f)
    
    print(f"\n✅ Single Agent Complete!")
    print(f"   Products: {len(all_products)}")
    print(f"   Time: {summary['total_time_minutes']:.1f} min")
    print(f"   Avg/Query: {summary['avg_time_per_query']:.1f}s")
    
    return all_products, summary


def run_multi_agent_test() -> Tuple[List[Dict], Dict]:
    """Run comprehensive multi-agent test"""
    print("\n" + "=" * 70)
    print("   TEST 2: MULTI-AGENT (multi_agent_scraper.py)")
    print("=" * 70)
    
    metrics = ComprehensiveMetrics("Multi-Agent (Parallel)")
    
    power_monitor = None
    if POWER_MONITOR_AVAILABLE:
        power_monitor = PowerMonitor()
        power_monitor.start_monitoring()
    
    try:
        from multi_agent_scraper import (
            BrowserAgent, AmazonAgent, FlipkartAgent, CromaAgent, RelianceAgent
        )
    except ImportError as e:
        print(f"❌ Could not import multi_agent_scraper.py: {e}")
        return [], {}
    
    all_products = []
    metrics.start()
    query_count = 0
    
    for category, queries in CATEGORIES:
        print(f"\n📂 Category: {category}")
        
        for query in queries:
            query_count += 1
            print(f"\n   [{query_count}/{TOTAL_QUERIES}] Searching: {query}")
            
            query_start = time.time()
            sources = {'Amazon': 0, 'Flipkart': 0, 'Croma': 0, 'Reliance Digital': 0}
            
            try:
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
                
                amazon_products = []
                flipkart_products = []
                croma_products = []
                reliance_products = []
                
                def run_amazon():
                    nonlocal amazon_products
                    amazon_products = amazon_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                def run_flipkart():
                    nonlocal flipkart_products
                    flipkart_products = flipkart_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                def run_croma():
                    nonlocal croma_products
                    croma_products = croma_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                def run_reliance():
                    nonlocal reliance_products
                    reliance_products = reliance_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                threads = [
                    Thread(target=run_amazon),
                    Thread(target=run_flipkart),
                    Thread(target=run_croma),
                    Thread(target=run_reliance)
                ]
                
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                
                amazon_browser.stop()
                flipkart_browser.stop()
                croma_browser.stop()
                reliance_browser.stop()
                
                query_time = time.time() - query_start
                products = amazon_products + flipkart_products + croma_products + reliance_products
                
                for p in products:
                    p['category'] = category
                    p['search_query'] = query
                    p['scrape_time'] = query_time
                
                sources['Amazon'] = len(amazon_products)
                sources['Flipkart'] = len(flipkart_products)
                sources['Croma'] = len(croma_products)
                sources['Reliance Digital'] = len(reliance_products)
                
                all_products.extend(products)
                metrics.add_query_result(query, category, products, query_time, sources)
                
                print(f"      ✓ Found {len(products)} products in {query_time:.1f}s (parallel)")
                print(f"        A:{len(amazon_products)} F:{len(flipkart_products)} C:{len(croma_products)} R:{len(reliance_products)}")
                
            except Exception as e:
                query_time = time.time() - query_start
                metrics.add_error(query, str(e))
                metrics.add_query_result(query, category, [], query_time, sources)
                print(f"      ❌ Error: {e}")
                
                try:
                    amazon_browser.stop()
                    flipkart_browser.stop()
                    croma_browser.stop()
                    reliance_browser.stop()
                except:
                    pass
    
    metrics.end()
    
    if power_monitor:
        try:
            power_monitor.record_measurement("Multi Agent Complete")
            power_report = power_monitor.generate_report()
            metrics.power_data = {
                'avg_cpu_percent': power_report.get('resource_utilization', {}).get('average_cpu_usage_percent', 0),
                'avg_cpu_power_watts': power_report.get('power_consumption', {}).get('average_cpu_power_watts', 0),
                'avg_memory_percent': power_report.get('resource_utilization', {}).get('average_memory_usage_percent', 0),
                'total_energy_kwh': power_report.get('energy_consumption', {}).get('total_energy_kwh', 0),
                'co2_emissions_grams': power_report.get('co2_emissions_grams', {}).get('india', 0),
            }
            print(f"\n⚡ Power: {metrics.power_data['avg_cpu_power_watts']:.2f}W, {metrics.power_data['total_energy_kwh']:.6f} kWh")
        except Exception as e:
            print(f"⚠️ Power monitoring error: {e}")
    
    if UMAP_ANALYZER_AVAILABLE and len(all_products) >= 10:
        try:
            print("\n🗺️ Running UMAP clustering...")
            analyzer = UMAPAnalyzer(all_products)
            analyzer.prepare_features()
            analyzer.run_umap()
            umap_metrics = analyzer.calculate_clustering_metrics()
            metrics.umap_data = {
                'silhouette_score': umap_metrics.get('silhouette_score', 0),
                'davies_bouldin_index': umap_metrics.get('davies_bouldin_index', 0),
                'cluster_purity': umap_metrics.get('cluster_purity', 0),
                'n_clusters': umap_metrics.get('n_clusters', 0)
            }
            analyzer.create_visualization('umap_multi_agent.png')
            print(f"✓ UMAP saved: umap_multi_agent.png")
        except Exception as e:
            print(f"⚠️ UMAP error: {e}")
    
    # Add sentiment analysis and price_numeric for multi-agent products
    all_products = add_price_numeric(all_products)
    all_products = add_sentiment_to_products(all_products)
    
    summary = metrics.get_summary()
    with open('scraped_multi_agent.pkl', 'wb') as f:
        pickle.dump({
            'products': all_products,
            'metrics': summary,
            'scraped_at': datetime.now().isoformat()
        }, f)
    
    print(f"\n✅ Multi-Agent Complete!")
    print(f"   Products: {len(all_products)}")
    print(f"   Time: {summary['total_time_minutes']:.1f} min")
    print(f"   Avg/Query: {summary['avg_time_per_query']:.1f}s")
    
    return all_products, summary


def select_test_configuration():
    """Select which category configuration to use"""
    global CATEGORIES, TOTAL_QUERIES, PRODUCTS_PER_QUERY
    
    print("\n" + "=" * 70)
    print("   COMPREHENSIVE COMPARISON TEST")
    print("   Single Agent (Try.py) vs Multi-Agent (multi_agent_scraper.py)")
    print("=" * 70)
    
    print("\n📋 Select Test Configuration:")
    print("   1. Standard Test (10 categories, 4 queries each) ~2-4 hours")
    print("   2. Large Categories Test (5 categories, 15-24 queries each) ~4-6 hours")
    print("   3. Quick Test (2 categories, 2 queries each) ~15-30 min")
    print("   4. Custom - Smartphones Only (24 queries)")
    print("   5. Custom - Laptops Only (22 queries)")
    
    config_choice = input("\nConfiguration (1-5): ").strip()
    
    if config_choice == '1':
        CATEGORIES = CATEGORIES_STANDARD
        print("\n✓ Selected: Standard Test")
    elif config_choice == '2':
        CATEGORIES = CATEGORIES_LARGE
        print("\n✓ Selected: Large Categories Test")
    elif config_choice == '3':
        CATEGORIES = CATEGORIES_QUICK
        PRODUCTS_PER_QUERY = 2
        print("\n✓ Selected: Quick Test")
    elif config_choice == '4':
        CATEGORIES = [("Smartphones", CATEGORIES_LARGE[0][1])]
        PRODUCTS_PER_QUERY = 2
        print("\n✓ Selected: Smartphones Only")
    elif config_choice == '5':
        CATEGORIES = [("Laptops", CATEGORIES_LARGE[1][1])]
        PRODUCTS_PER_QUERY = 2
        print("\n✓ Selected: Laptops Only")
    else:
        CATEGORIES = CATEGORIES_STANDARD
        print("\n✓ Using default: Standard Test")
    
    total_queries = sum(len(queries) for _, queries in CATEGORIES)
    TOTAL_QUERIES = total_queries
    
    print(f"\n📊 Configuration Summary:")
    print(f"   • Categories: {len(CATEGORIES)}")
    for cat, queries in CATEGORIES:
        print(f"     - {cat}: {len(queries)} queries")
    print(f"   • Total Queries: {total_queries}")
    print(f"   • Products/Query: {PRODUCTS_PER_QUERY}")
    print(f"   • Expected Products: ~{total_queries * PRODUCTS_PER_QUERY * 4} (max)")
    
    return CATEGORIES, TOTAL_QUERIES, PRODUCTS_PER_QUERY


def run_multi_agent_parallel_enhanced(category: str, query: str, products_per_query: int) -> Dict:
    """Run multi-agent scraping with parallel browser instances (like Try.py)"""
    from multi_agent_scraper import (
        BrowserAgent, AmazonAgent, FlipkartAgent, CromaAgent, RelianceAgent
    )
    
    results = {
        'Amazon': [],
        'Flipkart': [],
        'Croma': [],
        'Reliance Digital': []
    }
    
    def scrape_amazon():
        try:
            browser = BrowserAgent()
            browser.start()
            agent = AmazonAgent(browser)
            results['Amazon'] = agent.search(query, products_per_query, True)
            browser.stop()
        except Exception as e:
            print(f"      Amazon error: {e}")
    
    def scrape_flipkart():
        try:
            browser = BrowserAgent()
            browser.start()
            agent = FlipkartAgent(browser)
            results['Flipkart'] = agent.search(query, products_per_query, True)
            browser.stop()
        except Exception as e:
            print(f"      Flipkart error: {e}")
    
    def scrape_croma():
        try:
            browser = BrowserAgent()
            browser.start()
            agent = CromaAgent(browser)
            results['Croma'] = agent.search(query, products_per_query, True)
            browser.stop()
        except Exception as e:
            print(f"      Croma error: {e}")
    
    def scrape_reliance():
        try:
            browser = BrowserAgent()
            browser.start()
            agent = RelianceAgent(browser)
            results['Reliance Digital'] = agent.search(query, products_per_query, True)
            browser.stop()
        except Exception as e:
            print(f"      Reliance error: {e}")
    
    # Run all scrapers in parallel
    threads = [
        Thread(target=scrape_amazon),
        Thread(target=scrape_flipkart),
        Thread(target=scrape_croma),
        Thread(target=scrape_reliance)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)  # 2 minute timeout per query
    
    return results


def main():
    print(f"\n⏱️  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Select test configuration
    categories, total_queries, products_per_query = select_test_configuration()
    
    # Update global variables
    global CATEGORIES, TOTAL_QUERIES, PRODUCTS_PER_QUERY
    CATEGORIES = categories
    TOTAL_QUERIES = total_queries
    PRODUCTS_PER_QUERY = products_per_query
    
    print("\n📋 Select Test Mode:")
    print("   1. Single Agent only (Try.py)")
    print("   2. Multi-Agent only (multi_agent_scraper.py)")
    print("   3. Both (Full Comparison)")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        run_single_agent_test()
    
    if choice in ['2', '3']:
        run_multi_agent_test()
    
    print("\n" + "=" * 70)
    print("   TEST COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Generated Files:")
    for f in ['scraped_single_agent.pkl', 'scraped_multi_agent.pkl', 
              'umap_single_agent.png', 'umap_multi_agent.png']:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"   ✓ {f} ({size/1024:.1f} KB)")
    
    print(f"\n📊 Run 'python analyze_pkl_results.py' for detailed Chapter 5 report")
    print(f"\n⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
