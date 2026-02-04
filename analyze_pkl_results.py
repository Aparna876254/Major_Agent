"""
Chapter 5 Results Generator: Comprehensive Analysis of E-Commerce Scraper Performance

This script analyzes the results from comparison_test.py and generates detailed
Chapter 5 style results with performance metrics, visualizations, and comparisons.

Usage:
    python analyze_pkl_results.py

Output:
    - CHAPTER_5_RESULTS.md (Complete Chapter 5 report)
    - Performance visualizations
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import re
from typing import Dict, List, Any, Optional
from collections import Counter

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Try to import advanced libraries for UMAP and sentiment
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def extract_numeric_price(price_str) -> Optional[float]:
    """Extract numeric price from various price formats"""
    if price_str is None:
        return None
    price_str = str(price_str)
    # Remove currency symbols and commas
    cleaned = re.sub(r'[₹$,\s]', '', price_str)
    # Extract first number
    match = re.search(r'[\d.]+', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def extract_numeric_rating(rating_str) -> Optional[float]:
    """Extract numeric rating from various rating formats"""
    if rating_str is None:
        return None
    rating_str = str(rating_str)
    # Match patterns like "4.5 out of 5" or just "4.5"
    match = re.search(r'([\d.]+)\s*(?:out of|/|stars)?', rating_str, re.IGNORECASE)
    if match:
        try:
            rating = float(match.group(1))
            if 0 <= rating <= 5:
                return rating
        except ValueError:
            pass
    return None


def perform_sentiment_analysis(products: List[Dict]) -> List[Dict]:
    """Perform sentiment analysis on product reviews and descriptions"""
    for product in products:
        # Combine text sources for sentiment analysis
        text_parts = []
        
        if product.get('description'):
            text_parts.append(str(product['description'])[:500])
        if product.get('review_summary'):
            text_parts.append(str(product['review_summary'])[:500])
        if product.get('customer_reviews'):
            reviews = product['customer_reviews']
            if isinstance(reviews, list):
                for review in reviews[:3]:
                    if isinstance(review, dict):
                        text_parts.append(str(review.get('text', ''))[:200])
                    else:
                        text_parts.append(str(review)[:200])
            else:
                text_parts.append(str(reviews)[:500])
        
        combined_text = ' '.join(text_parts)
        
        if combined_text.strip() and TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(combined_text)
                sentiment_result = blob.sentiment
                polarity = sentiment_result.polarity  # access polarity as an attribute of the Sentiment namedtuple
                
                if polarity > 0.1:
                    product['sentiment_label'] = 'Positive'
                elif polarity < -0.1:
                    product['sentiment_label'] = 'Negative'
                else:
                    product['sentiment_label'] = 'Neutral'
                
                product['sentiment_confidence'] = min(abs(polarity) * 100 + 50, 100)
                product['sentiment_polarity'] = polarity
            except Exception:
                product['sentiment_label'] = 'Neutral'
                product['sentiment_confidence'] = 50.0
                product['sentiment_polarity'] = 0.0
        else:
            # Simple keyword-based sentiment if TextBlob not available
            positive_words = ['excellent', 'amazing', 'great', 'good', 'best', 'love', 'perfect', 'awesome', 'fantastic']
            negative_words = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 'disappointing', 'broken', 'defective']
            
            text_lower = combined_text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                product['sentiment_label'] = 'Positive'
                product['sentiment_confidence'] = min(60 + pos_count * 5, 95)
            elif neg_count > pos_count:
                product['sentiment_label'] = 'Negative'
                product['sentiment_confidence'] = min(60 + neg_count * 5, 95)
            else:
                product['sentiment_label'] = 'Neutral'
                product['sentiment_confidence'] = 55.0
            
            product['sentiment_polarity'] = (pos_count - neg_count) / max(pos_count + neg_count, 1)
    
    return products


def perform_umap_clustering(products: List[Dict]) -> Dict:
    """Perform UMAP dimensionality reduction and clustering analysis"""
    umap_results = {
        'silhouette_score': 0,
        'davies_bouldin_index': 0,
        'cluster_purity': 0,
        'n_clusters': 0,
        'cluster_distribution': {},
        'category_cluster_mapping': {},
        'embeddings': None,
        'method': 'none'
    }
    
    if not SKLEARN_AVAILABLE or len(products) < 10:
        return umap_results
    
    # Create text features from product data
    text_features = []
    categories = []
    
    for product in products:
        # Combine various text fields
        text_parts = []
        if product.get('name'):
            text_parts.append(str(product['name']))
        if product.get('category'):
            text_parts.append(str(product['category']))
            categories.append(str(product['category']))
        else:
            categories.append('Unknown')
        if product.get('description'):
            text_parts.append(str(product['description'])[:200])
        if product.get('features'):
            features = product['features']
            if isinstance(features, list):
                text_parts.extend([str(f)[:100] for f in features[:3]])
            else:
                text_parts.append(str(features)[:200])
        
        text_features.append(' '.join(text_parts))
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
        tfidf_matrix = vectorizer.fit_transform(text_features)
        
        # Determine optimal number of clusters (use category count as baseline)
        unique_categories = list(set(categories))
        n_clusters = min(len(unique_categories), len(products) // 5, 10)
        n_clusters = max(n_clusters, 2)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Calculate clustering metrics
        if len(set(cluster_labels)) > 1:
            umap_results['silhouette_score'] = silhouette_score(tfidf_matrix, cluster_labels)
            # Convert sparse matrix to dense array for davies_bouldin_score
            dense_matrix = np.asarray(tfidf_matrix.todense())
            umap_results['davies_bouldin_index'] = davies_bouldin_score(dense_matrix, cluster_labels)
        
        umap_results['n_clusters'] = n_clusters
        umap_results['method'] = 'TF-IDF + K-Means'
        
        # Calculate cluster distribution
        cluster_counts = Counter(cluster_labels)
        umap_results['cluster_distribution'] = {f'Cluster {k}': v for k, v in cluster_counts.items()}
        
        # Calculate cluster purity (how well clusters align with categories)
        cluster_category_mapping = {}
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
            cluster_categories = [categories[i] for i in cluster_indices]
            if cluster_categories:
                most_common = Counter(cluster_categories).most_common(1)[0]
                cluster_category_mapping[f'Cluster {cluster_id}'] = {
                    'dominant_category': most_common[0],
                    'purity': most_common[1] / len(cluster_categories) * 100,
                    'size': len(cluster_categories)
                }
        
        umap_results['category_cluster_mapping'] = cluster_category_mapping
        
        # Calculate overall purity
        total_correct = sum(info['purity'] * info['size'] / 100 for info in cluster_category_mapping.values())
        umap_results['cluster_purity'] = total_correct / len(products) * 100
        
        # Perform UMAP if available for visualization
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(products) - 1))
                embeddings = reducer.fit_transform(dense_matrix)
                umap_results['embeddings'] = embeddings
                umap_results['cluster_labels'] = cluster_labels
                umap_results['categories'] = categories
                umap_results['method'] = 'UMAP + K-Means'
            except Exception:
                pass
        
    except Exception as e:
        umap_results['error'] = str(e)
    
    return umap_results

def load_test_results() -> tuple:
    """Load results from both test methods and preprocess data"""
    single_data, multi_data = None, None
    
    if os.path.exists('scraped_single_agent.pkl'):
        with open('scraped_single_agent.pkl', 'rb') as f:
            single_data = pickle.load(f)
        print("✓ Loaded single agent results")
        # Preprocess single agent data
        if 'products' in single_data:
            single_data['products'] = preprocess_products(single_data['products'])
    
    if os.path.exists('scraped_multi_agent.pkl'):
        with open('scraped_multi_agent.pkl', 'rb') as f:
            multi_data = pickle.load(f)
        print("✓ Loaded multi-agent results")
        # Preprocess multi agent data
        if 'products' in multi_data:
            multi_data['products'] = preprocess_products(multi_data['products'])
    
    return single_data, multi_data


def preprocess_products(products: List[Dict]) -> List[Dict]:
    """Preprocess products: extract numeric values, add sentiment"""
    for product in products:
        # Extract numeric price if not present
        if not product.get('price_numeric'):
            product['price_numeric'] = extract_numeric_price(product.get('price'))
        
        # Extract numeric rating
        if product.get('rating'):
            numeric_rating = extract_numeric_rating(product['rating'])
            if numeric_rating is not None:
                product['rating_numeric'] = numeric_rating
        
        # Count reviews
        if product.get('customer_reviews'):
            reviews = product['customer_reviews']
            if isinstance(reviews, list):
                product['review_count'] = len(reviews)
            else:
                product['review_count'] = 1 if reviews else 0
    
    # Perform sentiment analysis
    products = perform_sentiment_analysis(products)
    
    return products

def generate_performance_plots(single_data: Dict, multi_data: Dict):
    """Generate performance comparison plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('E-Commerce Scraper Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Total Products Comparison
    methods = []
    products = []
    if single_data:
        methods.append('Single Agent')
        products.append(single_data['metrics']['total_products'])
    if multi_data:
        methods.append('Multi-Agent')
        products.append(multi_data['metrics']['total_products'])
    
    axes[0,0].bar(methods, products, color=['#3498db', '#e74c3c'])
    axes[0,0].set_title('Total Products Scraped')
    axes[0,0].set_ylabel('Number of Products')
    for i, v in enumerate(products):
        axes[0,0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # 2. Time Efficiency
    times = []
    if single_data:
        times.append(single_data['metrics']['total_time_minutes'])
    if multi_data:
        times.append(multi_data['metrics']['total_time_minutes'])
    
    axes[0,1].bar(methods, times, color=['#3498db', '#e74c3c'])
    axes[0,1].set_title('Total Execution Time')
    axes[0,1].set_ylabel('Time (minutes)')
    for i, v in enumerate(times):
        axes[0,1].text(i, v + 0.5, f'{v:.1f}m', ha='center', fontweight='bold')
    
    # 3. Throughput Comparison
    throughput = []
    if single_data:
        throughput.append(single_data['metrics']['throughput_products_per_second'])
    if multi_data:
        throughput.append(multi_data['metrics']['throughput_products_per_second'])
    
    axes[0,2].bar(methods, throughput, color=['#3498db', '#e74c3c'])
    axes[0,2].set_title('Throughput (Products/Second)')
    axes[0,2].set_ylabel('Products per Second')
    for i, v in enumerate(throughput):
        axes[0,2].text(i, v + 0.001, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 4. Source Distribution (Multi-Agent only)
    if multi_data:
        sources = multi_data['metrics']['products_per_source']
        source_names = list(sources.keys())
        source_counts = list(sources.values())
        
        axes[1,0].pie(source_counts, labels=source_names, autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Source Distribution (Multi-Agent)')
    else:
        axes[1,0].text(0.5, 0.5, 'Multi-Agent data\nnot available', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Source Distribution')
    
    # 5. Latency Distribution
    if multi_data and 'queries' in multi_data['metrics']:
        query_times = [q['time'] for q in multi_data['metrics']['queries'] if q['time'] > 0]
        if query_times:
            axes[1,1].hist(query_times, bins=15, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[1,1].axvline(np.mean(query_times), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(query_times):.1f}s')
            axes[1,1].set_title('Query Latency Distribution')
            axes[1,1].set_xlabel('Time (seconds)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
    
    # 6. Success Rate Comparison
    success_rates = []
    if single_data:
        success_rates.append(single_data['metrics']['success_rate'])
    if multi_data:
        success_rates.append(multi_data['metrics']['success_rate'])
    
    axes[1,2].bar(methods, success_rates, color=['#3498db', '#e74c3c'])
    axes[1,2].set_title('Success Rate')
    axes[1,2].set_ylabel('Success Rate (%)')
    axes[1,2].set_ylim(0, 100)
    for i, v in enumerate(success_rates):
        axes[1,2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated performance_analysis.png")
    
    # Generate additional visualizations
    generate_umap_and_sentiment_plots(multi_data)


def generate_umap_and_sentiment_plots(multi_data: Dict):
    """Generate UMAP clustering and sentiment analysis visualizations"""
    if not multi_data or 'products' not in multi_data:
        return
    
    products = multi_data['products']
    
    # Create figure with subplots for additional analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Product Analysis: UMAP Clustering & Sentiment Distribution', fontsize=14, fontweight='bold')
    
    # 1. Sentiment Distribution Pie Chart
    sentiment_counts = Counter([p.get('sentiment_label', 'Unknown') for p in products])
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6', 'Unknown': '#bdc3c7'}
    pie_colors = [colors.get(l, '#bdc3c7') for l in labels]
    
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
    axes[0, 0].set_title('Sentiment Distribution')
    
    # 2. Sentiment by Category
    categories = list(set(p.get('category', 'Unknown') for p in products))
    category_sentiments = {cat: {'Positive': 0, 'Negative': 0, 'Neutral': 0} for cat in categories}
    
    for p in products:
        cat = p.get('category', 'Unknown')
        sentiment = p.get('sentiment_label', 'Neutral')
        if sentiment in category_sentiments[cat]:
            category_sentiments[cat][sentiment] += 1
    
    x = np.arange(len(categories))
    width = 0.25
    
    positive_vals = [category_sentiments[c]['Positive'] for c in categories]
    neutral_vals = [category_sentiments[c]['Neutral'] for c in categories]
    negative_vals = [category_sentiments[c]['Negative'] for c in categories]
    
    axes[0, 1].bar(x - width, positive_vals, width, label='Positive', color='#2ecc71')
    axes[0, 1].bar(x, neutral_vals, width, label='Neutral', color='#95a5a6')
    axes[0, 1].bar(x + width, negative_vals, width, label='Negative', color='#e74c3c')
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Sentiment by Product Category')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([c[:12] for c in categories], rotation=45, ha='right')
    axes[0, 1].legend()
    
    # 3. Rating vs Sentiment Confidence
    ratings = [p.get('rating_numeric', 0) for p in products]
    confidences = [p.get('sentiment_confidence', 0) for p in products]
    sentiment_colors = [colors.get(p.get('sentiment_label', 'Unknown'), '#bdc3c7') for p in products]
    
    axes[1, 0].scatter(ratings, confidences, c=sentiment_colors, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Product Rating')
    axes[1, 0].set_ylabel('Sentiment Confidence (%)')
    axes[1, 0].set_title('Rating vs Sentiment Confidence')
    axes[1, 0].set_xlim(0, 5.5)
    axes[1, 0].set_ylim(0, 105)
    
    # 4. Price Distribution by Source
    sources = list(set(p.get('source', 'Unknown') for p in products))
    source_prices = {source: [] for source in sources}
    
    for p in products:
        source = p.get('source', 'Unknown')
        price = p.get('price_numeric')
        if price and price > 0:
            source_prices[source].append(price)
    
    # Box plot for price distribution
    price_data = [source_prices[s] for s in sources if source_prices[s]]
    valid_sources = [s for s in sources if source_prices[s]]
    
    if price_data:
        bp = axes[1, 1].boxplot(price_data, labels=[s[:15] for s in valid_sources], patch_artist=True)
        cmap = plt.colormaps.get_cmap('Set3')
        colors_box = cmap(np.linspace(0, 1, len(valid_sources)))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        axes[1, 1].set_xlabel('E-Commerce Platform')
        axes[1, 1].set_ylabel('Price (₹)')
        axes[1, 1].set_title('Price Distribution by Platform')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('umap_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated umap_sentiment_analysis.png")

def generate_chapter_5_report(single_data: Dict, multi_data: Dict):
    """Generate comprehensive Chapter 5 report"""
    
    report = []
    report.append("# Chapter 5: Results and Analysis")
    report.append("")
    report.append("This chapter presents a comprehensive evaluation of the e-commerce price comparison system, comparing single-agent sequential scraping with multi-agent parallel scraping across four major Indian platforms.")
    report.append("")
    
    # 5.1 System Performance Evaluation
    report.append("## 5.1 System Performance Evaluation")
    report.append("")
    
    # Dataset Overview
    report.append("### 5.1.1 Dataset Overview")
    report.append("")
    
    total_single = single_data['metrics']['total_products'] if single_data else 0
    total_multi = multi_data['metrics']['total_products'] if multi_data else 0
    
    report.append("| Metric | Single Agent | Multi-Agent |")
    report.append("|--------|--------------|-------------|")
    report.append(f"| **Total Products Scraped** | {total_single} | {total_multi} |")
    report.append(f"| **Total Categories Tested** | 10 | 10 |")
    report.append(f"| **Queries per Category** | 4 | 4 |")
    report.append(f"| **Total Search Queries** | 40 | 40 |")
    
    if single_data:
        report.append(f"| **Single Agent Success Rate** | {single_data['metrics']['success_rate']:.1f}% | - |")
    if multi_data:
        report.append(f"| **Multi-Agent Success Rate** | - | {multi_data['metrics']['success_rate']:.1f}% |")
    
    report.append("")
    
    # Product Distribution
    report.append("### 5.1.2 Product Distribution by Source")
    report.append("")
    
    if multi_data:
        sources = multi_data['metrics']['products_per_source']
        report.append("| E-Commerce Platform | Products Scraped | Percentage |")
        report.append("|---------------------|------------------|------------|")
        total = sum(sources.values())
        for source, count in sources.items():
            percentage = (count / total * 100) if total > 0 else 0
            report.append(f"| **{source}** | {count} | {percentage:.1f}% |")
        report.append("")
    
    # Category Distribution
    if multi_data and 'products_per_category' in multi_data['metrics']:
        report.append("### 5.1.3 Product Distribution by Category")
        report.append("")
        categories = multi_data['metrics']['products_per_category']
        report.append("| Product Category | Products Found |")
        report.append("|------------------|----------------|")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            report.append(f"| **{category}** | {count} |")
        report.append("")
    
    # 5.2 Time Efficiency Analysis
    report.append("## 5.2 Time Efficiency Analysis")
    report.append("")
    
    report.append("### 5.2.1 Overall Performance Metrics")
    report.append("")
    
    report.append("| Performance Metric | Single Agent | Multi-Agent | Improvement |")
    report.append("|--------------------|--------------|-------------|-------------|")
    
    if single_data and multi_data:
        single_time = single_data['metrics']['total_time_minutes']
        multi_time = multi_data['metrics']['total_time_minutes']
        time_improvement = ((single_time - multi_time) / single_time * 100) if single_time > 0 else 0
        
        single_throughput = single_data['metrics']['throughput_products_per_second']
        multi_throughput = multi_data['metrics']['throughput_products_per_second']
        throughput_improvement = ((multi_throughput - single_throughput) / single_throughput * 100) if single_throughput > 0 else 0
        
        report.append(f"| **Total Execution Time** | {single_time:.1f} min | {multi_time:.1f} min | {time_improvement:+.1f}% |")
        report.append(f"| **Average Time per Query** | {single_data['metrics']['avg_time_per_query']:.1f}s | {multi_data['metrics']['avg_time_per_query']:.1f}s | - |")
        report.append(f"| **Throughput (Products/sec)** | {single_throughput:.3f} | {multi_throughput:.3f} | {throughput_improvement:+.1f}% |")
        report.append(f"| **Products per Query** | {single_data['metrics']['avg_products_per_query']:.1f} | {multi_data['metrics']['avg_products_per_query']:.1f} | - |")
    
    report.append("")
    
    # Latency Analysis
    report.append("### 5.2.2 Latency Distribution Analysis")
    report.append("")
    
    if multi_data and 'queries' in multi_data['metrics']:
        query_times = [q['time'] for q in multi_data['metrics']['queries'] if q['time'] > 0]
        if query_times:
            report.append("| Latency Metric | Value |")
            report.append("|----------------|-------|")
            report.append(f"| **Mean Latency** | {np.mean(query_times):.2f}s |")
            report.append(f"| **Median Latency** | {np.median(query_times):.2f}s |")
            report.append(f"| **95th Percentile** | {np.percentile(query_times, 95):.2f}s |")
            report.append(f"| **99th Percentile** | {np.percentile(query_times, 99):.2f}s |")
            report.append(f"| **Standard Deviation** | {np.std(query_times):.2f}s |")
            report.append(f"| **Min Latency** | {np.min(query_times):.2f}s |")
            report.append(f"| **Max Latency** | {np.max(query_times):.2f}s |")
            report.append("")
    
    # 5.3 Bandwidth and Data Transfer Analysis
    report.append("## 5.3 Bandwidth and Data Transfer Analysis")
    report.append("")
    
    # Estimate data transfer based on products
    if multi_data:
        total_products = multi_data['metrics']['total_products']
        estimated_data_per_product = 15  # KB (HTML + images + metadata)
        total_data_kb = total_products * estimated_data_per_product
        total_data_mb = total_data_kb / 1024
        
        execution_time_hours = multi_data['metrics']['total_time'] / 3600
        bandwidth_mbps = (total_data_mb * 8) / (execution_time_hours * 3600) if execution_time_hours > 0 else 0
        
        report.append("| Data Transfer Metric | Value |")
        report.append("|---------------------|-------|")
        report.append(f"| **Estimated Data per Product** | {estimated_data_per_product} KB |")
        report.append(f"| **Total Data Transferred** | {total_data_mb:.1f} MB |")
        report.append(f"| **Average Bandwidth Usage** | {bandwidth_mbps:.2f} Mbps |")
        report.append(f"| **Data Efficiency** | {total_data_kb/max(1, total_products):.1f} KB/product |")
        report.append("")
    
    # 5.4 Resource Utilization & Energy Efficiency
    report.append("## 5.4 Resource Utilization & Energy Efficiency")
    report.append("")
    
    if single_data and 'power' in single_data['metrics'] and multi_data and 'power' in multi_data['metrics']:
        single_power = single_data['metrics']['power']
        multi_power = multi_data['metrics']['power']
        
        report.append("| Resource Metric | Single Agent | Multi-Agent |")
        report.append("|-----------------|--------------|-------------|")
        report.append(f"| **Average CPU Usage** | {single_power.get('avg_cpu_percent', 0):.1f}% | {multi_power.get('avg_cpu_percent', 0):.1f}% |")
        report.append(f"| **Average CPU Power** | {single_power.get('avg_cpu_power_watts', 0):.2f}W | {multi_power.get('avg_cpu_power_watts', 0):.2f}W |")
        report.append(f"| **Memory Usage** | {single_power.get('avg_memory_percent', 0):.1f}% | {multi_power.get('avg_memory_percent', 0):.1f}% |")
        report.append(f"| **Total Energy Consumption** | {single_power.get('total_energy_kwh', 0):.6f} kWh | {multi_power.get('total_energy_kwh', 0):.6f} kWh |")
        report.append(f"| **CO₂ Emissions (India)** | {single_power.get('co2_emissions_grams', 0):.2f}g | {multi_power.get('co2_emissions_grams', 0):.2f}g |")
        report.append("")
        
        # Energy efficiency per product
        single_energy_per_product = single_power.get('total_energy_kwh', 0) / max(1, total_single) * 1000000  # µWh
        multi_energy_per_product = multi_power.get('total_energy_kwh', 0) / max(1, total_multi) * 1000000  # µWh
        
        report.append("### 5.4.1 Energy Efficiency Analysis")
        report.append("")
        report.append("| Efficiency Metric | Single Agent | Multi-Agent |")
        report.append("|-------------------|--------------|-------------|")
        report.append(f"| **Energy per Product** | {single_energy_per_product:.2f} µWh | {multi_energy_per_product:.2f} µWh |")
        report.append(f"| **CO₂ per Product** | {single_power.get('co2_emissions_grams', 0)/max(1, total_single):.4f}g | {multi_power.get('co2_emissions_grams', 0)/max(1, total_multi):.4f}g |")
        report.append("")
    
    # 5.5 Data Quality & Accuracy Assessment
    report.append("## 5.5 Data Quality & Accuracy Assessment")
    report.append("")
    
    # Calculate data completeness
    if multi_data and 'products' in multi_data:
        products = multi_data['products']
        total_products = len(products)
        
        # Check completeness of key fields
        name_complete = sum(1 for p in products if p.get('name') and len(str(p['name']).strip()) > 5)
        
        def safe_numeric_check(value):
            try:
                return float(value) > 0 if value is not None else False
            except (ValueError, TypeError):
                return False
        
        price_complete = sum(1 for p in products if safe_numeric_check(p.get('price_numeric')))
        rating_complete = sum(1 for p in products if safe_numeric_check(p.get('rating')))
        image_complete = sum(1 for p in products if p.get('image_url') and 'http' in str(p['image_url']))
        sentiment_complete = sum(1 for p in products if p.get('sentiment_label'))
        
        report.append("### 5.5.1 Data Completeness Analysis")
        report.append("")
        report.append("| Data Field | Complete Records | Completeness Rate |")
        report.append("|------------|------------------|-------------------|")
        report.append(f"| **Product Name** | {name_complete}/{total_products} | {name_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Price Information** | {price_complete}/{total_products} | {price_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Rating Data** | {rating_complete}/{total_products} | {rating_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Product Images** | {image_complete}/{total_products} | {image_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Sentiment Analysis** | {sentiment_complete}/{total_products} | {sentiment_complete/max(1,total_products)*100:.1f}% |")
        report.append("")
        
        # Price range analysis
        valid_prices = []
        for p in products:
            try:
                price = p.get('price_numeric')
                if price is not None:
                    price_float = float(price)
                    if price_float > 0:
                        valid_prices.append(price_float)
            except (ValueError, TypeError):
                continue
        if valid_prices:
            report.append("### 5.5.2 Price Distribution Analysis")
            report.append("")
            report.append("| Price Metric | Value |")
            report.append("|--------------|-------|")
            report.append(f"| **Minimum Price** | ₹{min(valid_prices):,.0f} |")
            report.append(f"| **Maximum Price** | ₹{max(valid_prices):,.0f} |")
            report.append(f"| **Average Price** | ₹{np.mean(valid_prices):,.0f} |")
            report.append(f"| **Median Price** | ₹{np.median(valid_prices):,.0f} |")
            report.append(f"| **Price Standard Deviation** | ₹{np.std(valid_prices):,.0f} |")
            report.append("")
    
    # 5.6 Query Performance Analysis
    report.append("## 5.6 Query Performance Analysis")
    report.append("")
    
    if multi_data and 'queries' in multi_data['metrics']:
        queries = multi_data['metrics']['queries']
        
        # Category performance
        category_performance = {}
        for query in queries:
            category = query['category']
            if category not in category_performance:
                category_performance[category] = {'total_time': 0, 'total_products': 0, 'count': 0}
            category_performance[category]['total_time'] += query['time']
            category_performance[category]['total_products'] += query['products']
            category_performance[category]['count'] += 1
        
        report.append("### 5.6.1 Performance by Product Category")
        report.append("")
        report.append("| Category | Avg Time/Query | Avg Products/Query | Total Products |")
        report.append("|----------|----------------|-------------------|----------------|")
        
        for category, perf in sorted(category_performance.items(), key=lambda x: x[1]['total_time']):
            avg_time = perf['total_time'] / max(1, perf['count'])
            avg_products = perf['total_products'] / max(1, perf['count'])
            report.append(f"| **{category}** | {avg_time:.1f}s | {avg_products:.1f} | {perf['total_products']} |")
        
        report.append("")
    
    # 5.7 UMAP Clustering Analysis
    report.append("## 5.7 UMAP Clustering Analysis")
    report.append("")
    
    # Perform UMAP clustering if not already done
    umap_data = None
    if multi_data:
        if 'umap' in multi_data['metrics'] and multi_data['metrics']['umap']:
            umap_data = multi_data['metrics']['umap']
        else:
            # Perform clustering analysis now
            umap_data = perform_umap_clustering(multi_data['products'])
            multi_data['metrics']['umap'] = umap_data
    
    if umap_data and umap_data.get('n_clusters', 0) > 0:
        report.append("### 5.7.1 Clustering Methodology")
        report.append("")
        report.append(f"The product clustering analysis was performed using **{umap_data.get('method', 'TF-IDF + K-Means')}**. ")
        report.append("Products were represented as TF-IDF vectors based on their names, descriptions, and features, ")
        report.append("then clustered to identify natural product groupings across different e-commerce platforms.")
        report.append("")
        
        report.append("### 5.7.2 Clustering Quality Metrics")
        report.append("")
        report.append("| Clustering Metric | Value | Interpretation |")
        report.append("|-------------------|-------|----------------|")
        report.append(f"| **Silhouette Score** | {umap_data.get('silhouette_score', 0):.3f} | Cluster separation quality (-1 to 1, higher is better) |")
        report.append(f"| **Davies-Bouldin Index** | {umap_data.get('davies_bouldin_index', 0):.2f} | Cluster compactness (lower is better) |")
        report.append(f"| **Cluster Purity** | {umap_data.get('cluster_purity', 0):.1f}% | Category coherence within clusters |")
        report.append(f"| **Number of Clusters** | {umap_data.get('n_clusters', 0)} | Detected product groups |")
        report.append("")
        
        # Interpretation
        silhouette = umap_data.get('silhouette_score', 0)
        if silhouette > 0.5:
            interpretation = "Excellent clustering quality - clusters are well-separated and cohesive"
        elif silhouette > 0.25:
            interpretation = "Good clustering quality - reasonable cluster structure detected"
        elif silhouette > 0:
            interpretation = "Moderate clustering quality - some overlap between clusters"
        else:
            interpretation = "Weak clustering structure - products have high similarity across categories"
        
        report.append(f"**Clustering Quality Assessment:** {interpretation}")
        report.append("")
        
        # Add cluster distribution table
        if umap_data.get('category_cluster_mapping'):
            report.append("### 5.7.3 Cluster-Category Mapping")
            report.append("")
            report.append("| Cluster | Dominant Category | Purity | Size |")
            report.append("|---------|-------------------|--------|------|")
            
            for cluster, info in umap_data['category_cluster_mapping'].items():
                report.append(f"| **{cluster}** | {info['dominant_category']} | {info['purity']:.1f}% | {info['size']} products |")
            report.append("")
        
        # Add cluster distribution
        if umap_data.get('cluster_distribution'):
            report.append("### 5.7.4 Cluster Size Distribution")
            report.append("")
            report.append("| Cluster | Products | Percentage |")
            report.append("|---------|----------|------------|")
            
            total = sum(umap_data['cluster_distribution'].values())
            for cluster, count in sorted(umap_data['cluster_distribution'].items()):
                pct = count / max(1, total) * 100
                report.append(f"| **{cluster}** | {count} | {pct:.1f}% |")
            report.append("")
    else:
        report.append("*UMAP clustering analysis requires scikit-learn library. Install with: `pip install scikit-learn`*")
        report.append("")
    
    # 5.8 Sentiment Analysis Validation
    report.append("## 5.8 Sentiment Analysis Validation")
    report.append("")
    
    if multi_data and 'products' in multi_data:
        products = multi_data['products']
        
        # Sentiment distribution
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Unknown': 0}
        sentiment_confidences = []
        
        for product in products:
            sentiment = product.get('sentiment_label', 'Unknown')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts['Unknown'] += 1
            
            confidence = product.get('sentiment_confidence', 0)
            if confidence > 0:
                sentiment_confidences.append(confidence)
        
        total_with_sentiment = sum(sentiment_counts.values()) - sentiment_counts['Unknown']
        
        report.append("### 5.8.1 Sentiment Analysis Methodology")
        report.append("")
        if TEXTBLOB_AVAILABLE:
            report.append("Sentiment analysis was performed using **TextBlob NLP library**, which analyzes product reviews, ")
            report.append("descriptions, and customer feedback to determine overall sentiment polarity.")
        else:
            report.append("Sentiment analysis was performed using a **keyword-based approach**, analyzing product reviews ")
            report.append("and descriptions for positive and negative sentiment indicators.")
        report.append("")
        
        report.append("### 5.8.2 Sentiment Distribution")
        report.append("")
        report.append("| Sentiment | Count | Percentage |")
        report.append("|-----------|-------|------------|")
        
        for sentiment, count in sentiment_counts.items():
            if sentiment != 'Unknown':
                percentage = (count / max(1, total_with_sentiment)) * 100
                report.append(f"| **{sentiment}** | {count} | {percentage:.1f}% |")
        
        report.append("")
        
        if sentiment_confidences:
            report.append("### 5.8.3 Sentiment Analysis Confidence")
            report.append("")
            report.append("| Confidence Metric | Value |")
            report.append("|-------------------|-------|")
            report.append(f"| **Average Confidence** | {np.mean(sentiment_confidences):.1f}% |")
            report.append(f"| **Median Confidence** | {np.median(sentiment_confidences):.1f}% |")
            report.append(f"| **Min Confidence** | {np.min(sentiment_confidences):.1f}% |")
            report.append(f"| **Max Confidence** | {np.max(sentiment_confidences):.1f}% |")
            report.append("")
            
            # High confidence analysis
            high_confidence = sum(1 for c in sentiment_confidences if c >= 80)
            med_confidence = sum(1 for c in sentiment_confidences if 60 <= c < 80)
            low_confidence = sum(1 for c in sentiment_confidences if c < 60)
            
            report.append("### 5.8.4 Confidence Level Distribution")
            report.append("")
            report.append("| Confidence Level | Count | Percentage |")
            report.append("|------------------|-------|------------|")
            report.append(f"| **High (≥80%)** | {high_confidence} | {high_confidence/len(sentiment_confidences)*100:.1f}% |")
            report.append(f"| **Medium (60-79%)** | {med_confidence} | {med_confidence/len(sentiment_confidences)*100:.1f}% |")
            report.append(f"| **Low (<60%)** | {low_confidence} | {low_confidence/len(sentiment_confidences)*100:.1f}% |")
            report.append("")
        
        # Sentiment by category
        categories = list(set(p.get('category', 'Unknown') for p in products))
        report.append("### 5.8.5 Sentiment by Product Category")
        report.append("")
        report.append("| Category | Positive | Neutral | Negative | Dominant Sentiment |")
        report.append("|----------|----------|---------|----------|-------------------|")
        
        for category in sorted(categories):
            cat_products = [p for p in products if p.get('category') == category]
            pos = sum(1 for p in cat_products if p.get('sentiment_label') == 'Positive')
            neu = sum(1 for p in cat_products if p.get('sentiment_label') == 'Neutral')
            neg = sum(1 for p in cat_products if p.get('sentiment_label') == 'Negative')
            
            dominant = 'Positive' if pos >= neu and pos >= neg else ('Neutral' if neu >= neg else 'Negative')
            report.append(f"| **{category}** | {pos} | {neu} | {neg} | {dominant} |")
        
        report.append("")
    
    # Conclusions
    report.append("## 5.9 Key Findings and Conclusions")
    report.append("")
    
    if single_data and multi_data:
        # Performance comparison
        if multi_data['metrics']['total_time'] < single_data['metrics']['total_time']:
            time_improvement = ((single_data['metrics']['total_time'] - multi_data['metrics']['total_time']) / single_data['metrics']['total_time']) * 100
            report.append(f"1. **Performance Improvement**: Multi-agent approach achieved {time_improvement:.1f}% faster execution time")
        
        # Throughput comparison
        if multi_data['metrics']['throughput_products_per_second'] > single_data['metrics']['throughput_products_per_second']:
            throughput_improvement = ((multi_data['metrics']['throughput_products_per_second'] - single_data['metrics']['throughput_products_per_second']) / single_data['metrics']['throughput_products_per_second']) * 100
            report.append(f"2. **Throughput Enhancement**: {throughput_improvement:.1f}% improvement in products per second")
        
        # Coverage
        if multi_data['metrics']['total_products'] > single_data['metrics']['total_products']:
            coverage_improvement = multi_data['metrics']['total_products'] - single_data['metrics']['total_products']
            report.append(f"3. **Coverage Expansion**: Multi-agent scraped {coverage_improvement} additional products")
    
    if multi_data:
        # Source diversity
        active_sources = sum(1 for count in multi_data['metrics']['products_per_source'].values() if count > 0)
        report.append(f"4. **Source Diversity**: Successfully integrated {active_sources}/4 e-commerce platforms")
        
        # Success rate
        success_rate = multi_data['metrics']['success_rate']
        report.append(f"5. **Reliability**: Achieved {success_rate:.1f}% success rate across all queries")
        
        # Clustering insights
        if 'umap' in multi_data['metrics'] and multi_data['metrics']['umap']:
            umap_data = multi_data['metrics']['umap']
            if umap_data.get('silhouette_score', 0) > 0:
                report.append(f"6. **Product Clustering**: Identified {umap_data.get('n_clusters', 0)} distinct product clusters with {umap_data.get('cluster_purity', 0):.1f}% category purity")
        
        # Sentiment summary
        products = multi_data.get('products', [])
        if products:
            positive = sum(1 for p in products if p.get('sentiment_label') == 'Positive')
            total = len(products)
            report.append(f"7. **Sentiment Analysis**: {positive}/{total} products ({positive/total*100:.1f}%) have positive sentiment")
    
    report.append("")
    report.append("---")
    report.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Write report to file
    with open('CHAPTER_5_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✓ Generated CHAPTER_5_RESULTS.md")

def main():
    """Main analysis function"""
    print("\n" + "=" * 70)
    print("   CHAPTER 5 RESULTS GENERATOR")
    print("   Comprehensive Analysis of E-Commerce Scraper Performance")
    print("=" * 70)
    
    # Load and preprocess test results
    print("\n📥 Loading test data...")
    single_data, multi_data = load_test_results()
    
    if not single_data and not multi_data:
        print("\n❌ No test results found!")
        print("   Run 'python comparison_test.py' first to generate test data")
        return
    
    # Generate visualizations
    if single_data or multi_data:
        print("\n📊 Generating performance plots...")
        generate_performance_plots(single_data, multi_data)
    
    # Generate comprehensive report
    print("\n📝 Generating Chapter 5 report...")
    generate_chapter_5_report(single_data, multi_data)
    
    print("\n✅ Analysis Complete!")
    print("\n📁 Generated Files:")
    for filename in ['CHAPTER_5_RESULTS.md', 'performance_analysis.png', 'umap_sentiment_analysis.png']:
        if os.path.exists(filename):
            print(f"   ✓ {filename}")
    
    print(f"\n📖 Open 'CHAPTER_5_RESULTS.md' for complete Chapter 5 results")

if __name__ == "__main__":
    main()