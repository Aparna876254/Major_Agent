# E-Commerce Price Comparison Tool with RAG & Neural Network Sentiment Analysis

## Project Documentation

**Author:** Anurag U  
**Repository:** AnuragU03/Major  
**Date:** January 2026

---

## Table of Contents

1. [Scope / Objectives of the Project](#1-scope--objectives-of-the-project)
2. [Methodology](#2-methodology)
3. [Technical Details](#3-technical-details)

---

## 1. Scope / Objectives of the Project

### 1.1 Primary Objective

To develop an **intelligent e-commerce price comparison system** that aggregates product information from multiple Indian e-commerce platforms (**Amazon.in, Flipkart, Croma, and Reliance Digital**) and provides users with comprehensive product analysis including pricing, specifications, reviews, and **Neural Network-powered sentiment analysis using DistilBERT**.

### 1.2 Specific Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Multi-Platform Product Aggregation** | Scrape and collect product data from **Amazon.in, Flipkart, Croma, and Reliance Digital** simultaneously for unified comparison |
| 2 | **RAG-Based Smart Caching** | Implement Retrieval-Augmented Generation (RAG) pipeline with semantic search to cache products locally and reduce redundant scraping |
| 3 | **Deep Product Information Extraction** | Extract comprehensive product details including technical specifications, features, customer reviews (with pagination up to 50 reviews), rating breakdowns, and descriptions |
| 4 | **Neural Network Sentiment Analysis** | Deploy DistilBERT transformer model fine-tuned on SST-2 with aspect-based sentiment analysis (quality, performance, battery, camera, display, value) |
| 5 | **Interactive Visualization** | Provide a rich GUI interface displaying products with images, prices, specs, sentiment scores, and direct purchase links |
| 6 | **Intelligent Product Filtering** | Automatically filter out accessories, irrelevant products, and validate product relevance to search queries |
| 7 | **Anti-Detection Measures** | Implement StealthBrowser with randomized user agents, human-like scrolling, and rate limiting to avoid bot detection |

### 1.3 Scope Boundaries

**In Scope:**
- Amazon.in, Flipkart, Croma, and Reliance Digital price comparison
- Product specifications, reviews, and ratings extraction
- Advanced review scraping with pagination (up to 50 reviews per product)
- Neural Network sentiment analysis using DistilBERT (Transformer-based)
- Aspect-based sentiment analysis (quality, performance, battery, camera, display, value)
- Support for multiple training datasets (Amazon Polarity, Amazon Reviews 2023, Yelp Reviews)
- Local RAG storage with semantic search
- Anti-detection measures (StealthBrowser, RateLimiter, human-like scrolling)
- Desktop GUI application (Tkinter)

**Out of Scope:**
- Mobile application
- Real-time price alerts/notifications
- User authentication or personalized pricing
- Other e-commerce platforms (Myntra, Snapdeal, etc.)

---

## 2. Methodology

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                    (Tkinter GUI Application)                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ Local Exact │──▶│ Fuzzy Match │──▶│ Web Scrape  │                  │
│  │   Search    │  │   (60%)     │  │  (Selenium) │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
     ┌─────────────────────────────────────────────────────────┐
     │                  MULTI-AGENT SCRAPERS                   │
     │  ┌───────────┐ ┌───────────┐ ┌───────┐ ┌───────────┐  │
     │  │ Amazon.in │ │ Flipkart  │ │ Croma │ │ Reliance  │  │
     │  │   Agent   │ │   Agent   │ │ Agent │ │   Agent   │  │
     │  └───────────┘ └───────────┘ └───────┘ └───────────┘  │
     └─────────────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│   ProductRAGStorage     │              │   Enhanced Sentiment Analyzer   │
│   - TF-IDF Vectorizer   │              │   - DistilBERT Transformer      │
│   - Cosine Similarity   │              │   - Aspect-Based Analysis       │
│   - Pickle Persistence  │              │   - GPU/CPU Auto-detection      │
└─────────────────────────┘              └─────────────────────────────┘
              │                                         │
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│  product_rag_db.pkl     │              │  Pre-trained Model (cached)     │
└─────────────────────────┘              └─────────────────────────────┘
```

### 2.2 Search Workflow (RAG Pipeline)

**Step-by-Step Process:**

```
User Query: "Samsung Galaxy Watch"
        │
        ▼
┌───────────────────────────────────┐
│ STEP 1: Local Exact Match         │  ◄── Fast (< 1 second)
│ Check cached products database    │
└───────────────────────────────────┘
        │ Not Found
        ▼
┌───────────────────────────────────┐
│ STEP 2: Fuzzy Matching (60%)      │  ◄── Semantic Search
│ TF-IDF + Cosine Similarity        │
│ Token overlap threshold: 60%      │
└───────────────────────────────────┘
        │ Not Found
        ▼
┌───────────────────────────────────┐
│ STEP 3: External Web Scraping     │  ◄── Parallel (4 threads)
│ Amazon.in + Flipkart + Croma +    │
│ Reliance Digital (StealthBrowser) │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 4: Validation & Filtering    │
│ - Remove accessories              │
│ - Validate brand matching         │
│ - Check price validity            │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 5: Sentiment Analysis        │
│ Enhanced multi-aspect analysis    │
│ Analyze reviews → Score products  │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 6: Auto-Storage in RAG DB    │
│ Cache for future searches         │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 7: Display in GUI            │
│ With images, specs, sentiment     │
└───────────────────────────────────┘
```

### 2.3 Sentiment Analysis Methodology

The system uses a **Neural Network-based sentiment analyzer** powered by DistilBERT, a transformer model pre-trained on SST-2 (Stanford Sentiment Treebank). **The key innovation is extracting actual customer review TEXT (not just star ratings) for authentic sentiment analysis.**

| Step | Process | Technical Details |
|------|---------|-------------------|
| **1. Model Loading** | Load pre-trained DistilBERT | `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace |
| **2. Review Extraction** | Scrape actual customer comments | Wait for page load (5s) + scroll to reviews section + extract text from `div.a6dZNm.mIW33x` (Flipkart) |
| **3. Text Collection** | Gather review text | Priority: Customer Reviews (up to 5) > Review Summary > Description |
| **4. Preprocessing** | Clean input text | Remove URLs, special chars, truncate to 512 tokens |
| **5. Inference** | Run through transformer | HuggingFace sentiment-analysis pipeline |
| **6. Score Mapping** | Convert to 3-class | POSITIVE/NEGATIVE with confidence threshold (0.6) |
| **7. Aggregation** | Combine multiple reviews | Average scores across all analyzed texts |
| **8. Explanation** | Generate summary | "Based on X reviews: Y positive, Z negative" |

#### Customer Review Extraction Strategy

The scraper uses a **3-method fallback approach** to reliably extract actual review comments:

```
Method 1: Direct CSS Selector Search
├── div.a6dZNm.mIW33x (Flipkart review text)
├── div.ZmyHeo (Alternative selector)
└── Multiple fallback selectors

Method 2: Container-Based Extraction
├── Find review containers (div.col.EPCmJX)
├── Extract text lines from each container
└── Filter for review content (>30 chars, <1000 chars)

Method 3: Regex Pattern Matching (Fallback)
├── Match patterns like "Fabulous! ...", "Awesome ..."
├── Extract text following review titles
└── Clean and validate extracted text
```

#### Supported Training Datasets (for fine-tuning)

| Dataset | Source | Size | Labels |
|---------|--------|------|--------|
| **Amazon Polarity** | `mteb/amazon_polarity` | 400K+ | Binary (Positive/Negative) |
| **Amazon Reviews 2023** | `McAuley-Lab/Amazon-Reviews-2023` | Millions | 1-5 Stars by Category |
| **Yelp Reviews** | `Yelp/yelp_review_full` | 650K+ | 1-5 Stars |

#### Model Architecture

```
DistilBERT (66M parameters)
├── 6 Transformer Layers
├── 768 Hidden Dimensions
├── 12 Attention Heads
├── Max Sequence Length: 512 tokens
└── Fine-tuned on SST-2 for Binary Sentiment
```

### 2.4 Web Scraping Methodology

**Anti-Bot Measures Implemented:**
- Random User-Agent rotation (multiple Chrome versions)
- WebDriver flags disabled (`navigator.webdriver = undefined`)
- Chrome automation extension disabled (`excludeSwitches: enable-automation`)
- Blink automation features disabled (`disable-blink-features=AutomationControlled`)
- Random delays between requests (3-10 seconds)
- Human-like scrolling with randomized distances (200-800 pixels)
- Window size randomization (1920x1080, 1366x768, 1440x900)
- RateLimiter class to prevent IP blocks (10 requests/minute)
- Session persistence for cookie management

**Smart Page Loading Strategy:**
- **Initial Wait**: 5 seconds for page to fully load
- **Progressive Scrolling**: 8 scroll steps with 0.8s pause each to trigger lazy loading
- **Reviews Section Wait**: Scroll to "Ratings & Reviews" heading + 2-3 second wait
- **Content Verification**: Print debug output to confirm data extraction

**Data Extraction Strategy:**
- Multiple CSS selector fallbacks for each field
- Tab management for product detail pages
- Retry logic (2 attempts with 30-second timeouts)
- Pagination support for deep review extraction (up to 50 reviews)
- 3-method fallback for customer reviews (direct → container → regex)

**Supported E-Commerce Platforms:**

| Platform | URL | Features |
|----------|-----|----------|
| Amazon.in | amazon.in | Full specs, reviews, rating breakdown, customer reviews |
| Flipkart | flipkart.com | Specs, reviews, highlights, category ratings (Camera/Battery/Display/Design), customer review text |
| Croma | croma.com | Product search, pricing, key features, specifications |
| Reliance Digital | reliancedigital.in | Product search (`/products?q=`), pricing, specs |

---

## 3. Technical Details

### 3.1 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Programming Language** | Python | 3.8+ |
| **Web Scraping** | Selenium WebDriver | 4.0.0+ |
| **Browser Automation** | Chrome + ChromeDriver | Auto-managed |
| **Neural Networks** | PyTorch + Transformers | 2.0+, 4.35+ |
| **Pre-trained Model** | DistilBERT (HuggingFace) | SST-2 fine-tuned |
| **Dataset Loading** | HuggingFace Datasets | 2.14+ |
| **Data Processing** | Pandas, NumPy | 1.3+, 1.21+ |
| **Machine Learning** | scikit-learn | 1.0+ |
| **GUI Framework** | Tkinter | Built-in |
| **Image Processing** | Pillow (PIL) | 9.0+ |
| **HTTP Requests** | Requests | 2.26+ |
| **Visualization** | Matplotlib, Seaborn | 3.5+, 0.11+ |
| **Dimensionality Reduction** | UMAP-learn | 0.5+ |
| **System Monitoring** | psutil | 5.8+ |

### 3.2 Key Classes & Modules

| Class/Module | File | Purpose |
|--------------|------|---------|
| `NeuralSentimentAnalyzer` | `neural_sentiment_analyzer.py` | DistilBERT-based sentiment analysis |
| `EnhancedSentimentAnalyzer` | `multi_agent_scraper.py` | Multi-model analyzer with aspect-based analysis |
| `DatasetLoader` | `neural_sentiment_analyzer.py` | Load Amazon/Yelp datasets from HuggingFace |
| `ProductRAGStorage` | `Try.py` | RAG-based product caching with TF-IDF semantic search |
| `RAGStorage` | `umap_rag_analyzer.py` | Generic RAG storage with TF-IDF vectorization |
| `UMAPAnalyzer` | `umap_rag_analyzer.py` | UMAP dimensionality reduction and clustering visualization |
| `ProductValidator` | `product_validator.py` | Brand/Series/Model validation with accessory filtering |
| `ProductValidation` | `product_validator.py` | Dataclass for validation results |
| `PowerMonitor` | `power_monitor.py` | CPU/Memory/Energy consumption tracking |
| `Measurement` | `power_monitor.py` | Dataclass for power measurements |
| `AdvancedReviewScraper` | `multi_agent_scraper.py` | Deep review extraction with pagination (50+ reviews) |
| `CromaScraper` | `multi_agent_scraper.py` | Croma.com product scraper |
| `RelianceDigitalScraper` | `multi_agent_scraper.py` | RelianceDigital.in product scraper |
| `StealthBrowser` | `multi_agent_scraper.py` | Anti-detection Chrome automation |
| `RateLimiter` | `multi_agent_scraper.py` | Request rate limiting to avoid IP blocks |
| `AmazonAgent` | `multi_agent_scraper.py` | Dedicated agent for Amazon.in scraping |
| `FlipkartAgent` | `multi_agent_scraper.py` | Dedicated agent for Flipkart scraping |
| `CromaAgent` | `multi_agent_scraper.py` | Dedicated agent for Croma.com scraping |
| `RelianceAgent` | `multi_agent_scraper.py` | Dedicated agent for RelianceDigital.in scraping |
| `SentimentAgent` | `multi_agent_scraper.py` | Agent wrapper for neural sentiment analysis |
| `FilterAgent` | `multi_agent_scraper.py` | Product filtering and validation |
| `GUIAgent` | `multi_agent_scraper.py` | Results display agent |

### 3.3 Neural Network Model Specifications

**DistilBERT Sentiment Classifier:**

```python
# Model from HuggingFace
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Pipeline Configuration
pipeline(
    "sentiment-analysis",
    model=model_name,
    device=0 if cuda_available else -1,  # GPU/CPU auto-detection
    truncation=True,
    max_length=512
)

# Output Format
{
    'label': 'POSITIVE' or 'NEGATIVE',
    'score': 0.0 to 1.0  # Confidence
}
```

**Sentiment Score Mapping:**

```python
# Convert binary to 3-class sentiment
if label == 'POSITIVE' and score > 0.6:
    sentiment = 'positive'
elif label == 'NEGATIVE' and score > 0.6:
    sentiment = 'negative'
else:
    sentiment = 'neutral'

# Score normalization (0 = negative, 0.5 = neutral, 1 = positive)
final_score = 0.5 + (score * 0.5) if positive else 0.5 - (score * 0.5)
```

**Semantic Search (RAG) Configuration:**

```python
# Vectorization
TfidfVectorizer(max_features=500)

# Similarity Computation
cosine_similarity(query_vector, document_vectors)
```

### 3.4 Data Extracted Per Product

| Field | Source | Description |
|-------|--------|-------------|
| `name` | Search page | Product title |
| `price` | Search/Product page | Price in INR |
| `rating` | Both pages | Star rating (e.g., "4.6") |
| `reviews` | Both pages | Review count |
| `image_url` | Search page | Product thumbnail |
| `product_link` | Search page | Direct URL to product |
| `technical_details` | Product page | Dict of specifications |
| `features` | Product page | List of feature bullets |
| `description` | Product page | Full product description |
| `customer_reviews` | Product page | **List of actual review text** (up to 5 reviews with comment text, not just stars) |
| `rating_breakdown` | Product page | Count per star level (5★: 2,12,393) |
| `category_ratings` | Flipkart | Camera, Battery, Display, Design ratings |
| `review_summary` | Product page | "Customers say" summary |
| `sentiment` | ML Model | positive/neutral/negative |
| `sentiment_score` | ML Model | 0.0 to 1.0 |
| `sentiment_source` | ML Model | "customer_reviews" or "description" |
| `sentiment_explanation` | ML Model | "Based on X reviews: Y positive, Z negative" |

#### Customer Reviews Format

```python
# Each review in customer_reviews list
{
    'text': 'So beautiful, so elegant, just a vowww 😍❤️',  # Actual comment text
    'title': 'Fabulous!',  # Review title (optional)
    'reviewer': 'Akshay Meena'  # Reviewer name (optional)
}
```

### 3.5 Project File Structure

```
Major Project/
├── Try.py                        # Main application entry point (RAG + GUI)
├── multi_agent_scraper.py        # Multi-agent web scraping system
├── neural_sentiment_analyzer.py  # Neural network sentiment analysis (DistilBERT)
├── requirements.txt              # Python dependencies
├── README.md                     # Quick start guide
├── PROJECT_DOCUMENTATION.md      # This documentation
├── product_rag_db.pkl            # RAG storage (auto-generated)
└── __pycache__/                  # Python cache
```

### 3.6 Dependencies (requirements.txt)

```
# Core Dependencies
selenium==4.38.0
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
Pillow==12.0.0
requests==2.32.5
webdriver-manager==4.0.2

# Neural Network Sentiment Analysis (DistilBERT)
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
```

### 3.6 Dependencies (requirements.txt)

```
# Core Dependencies
selenium==4.38.0
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
Pillow==12.0.0
requests==2.32.5
webdriver-manager==4.0.2

# Neural Network Sentiment Analysis (DistilBERT)
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
```

### 3.7 Performance Characteristics

| Metric | Value |
|--------|-------|
| First search (cold cache) | 30-60 seconds |
| Cached search (warm cache) | < 1 second |
| Products per search | 2-20 (configurable) |
| Storage per 100 products | ~1 MB |
| Neural model first load | 5-15 seconds (downloads ~260MB model) |
| Sentiment analysis per product | < 1 second |
| Model accuracy (SST-2) | ~91% |
| GPU acceleration | Automatic if CUDA available |

### 3.8 Algorithms Used

1. **Transformer Architecture (DistilBERT)**
   - 6-layer distilled version of BERT
   - Self-attention mechanism for context understanding
   - Pre-trained on large text corpus, fine-tuned on SST-2

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Used for text vectorization in RAG storage
   - Converts text to numerical feature vectors for similarity search

3. **Cosine Similarity**
   - Measures similarity between product descriptions
   - Used for semantic product matching in RAG pipeline

4. **Sentiment Aggregation**
   - Analyzes multiple reviews per product
   - Averages confidence scores for final sentiment

### 3.9 HuggingFace Datasets Integration

The system supports loading datasets for potential fine-tuning:

```python
from neural_sentiment_analyzer import DatasetLoader

# Load Amazon Polarity (binary sentiment)
amazon_data = DatasetLoader.load_amazon_polarity(sample_size=5000)

# Load Amazon Reviews 2023 (by category)
electronics = DatasetLoader.load_amazon_reviews_2023(category="Electronics")

# Load Yelp Reviews (5-star ratings)
yelp_data = DatasetLoader.load_yelp_reviews(sample_size=5000)

# Prepare combined dataset for fine-tuning
combined = DatasetLoader.prepare_combined_dataset(
    amazon_samples=5000,
    yelp_samples=5000
)
```

### 3.10 Error Handling Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Timeout Protection** | 30-second page load limits |
| **Element Not Found** | Multiple CSS selector fallbacks |
| **Tab Management** | Auto-cleanup of browser tabs on errors |
| **Data Validation** | Skips invalid products without crashing |
| **Model Fallback** | Returns 'unknown' sentiment if model not loaded |
| **GPU Fallback** | Automatic CPU fallback if CUDA unavailable |
| **Import Protection** | Graceful degradation if transformers not installed |

### 3.11 System Requirements

- **Operating System:** Windows/Linux/macOS
- **Python:** 3.8 or higher
- **Browser:** Google Chrome (latest version)
- **RAM:** 8GB minimum (16GB recommended for GPU)
- **Storage:** 1GB for application + model cache
- **GPU:** Optional (CUDA-compatible for faster inference)
- **Internet:** Required for web scraping and first model download

### 3.12 Installation & Usage

**Installation:**
```bash
# Clone repository
git clone https://github.com/AnuragU03/Major.git
cd "Major Project"

# Install dependencies
pip install -r requirements.txt

# Test neural sentiment analyzer (optional)
python neural_sentiment_analyzer.py

# Run the application
python Try.py
```

**Alternative: Multi-Agent Scraper**
```bash
# Run the multi-agent version
python multi_agent_scraper.py
```

**Usage:**
1. Launch the application with `python Try.py`
2. Enter product name (e.g., "Samsung Galaxy Watch")
3. Specify number of products per source
4. View results in interactive GUI with neural sentiment analysis

---

## Summary

This project implements a comprehensive e-commerce price comparison system that combines:

- **Web Scraping** (Selenium) for multi-platform data extraction (Amazon, Flipkart, Croma, Reliance Digital)
- **RAG Pipeline** (TF-IDF + Cosine Similarity) for smart caching
- **Neural Network Sentiment Analysis** (DistilBERT Transformer) for accurate review analysis
- **Aspect-Based Sentiment** for analyzing quality, performance, battery, camera, display, and value
- **Advanced Anti-Detection** (StealthBrowser, RateLimiter, human-like scrolling)
- **Multiple Dataset Support** (Amazon Polarity, Amazon Reviews 2023, Yelp Reviews)
- **GUI** (Tkinter) for interactive user experience

---

## 4. Testing & Validation

### 4.1 Test Framework

The project includes comprehensive testing and validation through multiple test scripts:

| Test Script | Purpose |
|-------------|---------|
| `comparison_test.py` | Compare single-agent vs multi-agent performance |
| `analyze_pkl_results.py` | Analyze scraped data from pickle files |
| `product_validator.py` | Validate product relevance and filter accessories |

### 4.2 Product Validation

The `ProductValidator` class ensures scraped products are relevant to the search query:

```python
class ProductValidator:
    """Validates products against search query using Brand/Series/Model matching"""
    
    def validate_product(self, product_name, search_query):
        # Extract brand, series, model from search query
        # Check if product name contains expected components
        # Filter out accessories (cases, covers, screen protectors)
        # Return: (is_valid, confidence_score, reason)
```

**Validation Rules:**
1. **Brand Matching**: Product must contain the brand from search query
2. **Series/Model Matching**: Must match series or model number
3. **Accessory Filtering**: Filters out cases, covers, screen protectors, chargers
4. **Truncation Check**: Detects truncated names (< 15 chars or exact brand match)

**Example Validation:**
```
Search Query: "iPhone 15 Pro"
✅ "Apple iPhone 15 Pro 256GB Black" → Valid (brand + series + model match)
❌ "iPhone 15 Pro Case Cover" → Invalid (accessory detected)
❌ "Apple" → Invalid (truncated name)
```

### 4.3 Comparison Testing (400 Products)

The `comparison_test.py` script performs comprehensive comparison between:
- **Single Agent** (`Try.py`): Sequential scraping
- **Multi-Agent** (`multi_agent_scraper.py`): Parallel scraping with 4 browser agents

**Test Parameters:**
- **10 Categories**: Smartphones, Laptops, Smartwatches, Tablets, Wireless Earbuds, Headphones, Smart TVs, Cameras, Gaming, Smart Speakers
- **40 Queries**: 4 queries per category
- **4 Sources**: Amazon.in, Flipkart, Croma, Reliance Digital
- **Target**: ~400 products total

**Metrics Collected:**
| Metric Category | Metrics |
|-----------------|---------|
| **Performance** | Total time, Avg time/query, Products/second |
| **Power** | CPU usage %, CPU power (W), Memory %, Energy (kWh), CO₂ emissions |
| **Accuracy** | Name/Price/Rating/Reviews completeness %, Overall accuracy score |
| **UMAP Clustering** | Silhouette score, Davies-Bouldin index, Cluster purity |

### 4.4 PKL Analysis

The `analyze_pkl_results.py` script provides detailed analysis of scraped data:

**Latency Analysis:**
```
• Average Latency: 66.58s per query
• Min Latency: 49.42s
• Max Latency: 108.31s
• P95 Latency: 81.38s
• P99 Latency: 101.18s
```

**Bandwidth Analysis:**
```
• Total Data Scraped: 674.30 KB
• Throughput: 0.0537 products/sec
• Avg Product Size: 4829 bytes
```

**Accuracy Analysis (Data Completeness):**
| Field | Single Agent | Multi-Agent |
|-------|--------------|-------------|
| Name | 100% | 100% |
| Price | 100% | 79.7% |
| Rating | 48.4% | 81.1% |
| Image URL | 100% | 100% |
| Product Link | 100% | 100% |
| Technical Details | 41.3% | 68.5% |
| **Overall Accuracy** | **87.1%** | **90.2%** |

**Website Performance Ranking:**
| Rank | Website | Score | Products | Data Quality |
|------|---------|-------|----------|--------------|
| #1 | Amazon.in | 75.2 | 38 | 100% price, 100% rating |
| #2 | Flipkart | 72.7 | 36 | 100% price, 94.4% rating |
| #3 | Croma | 60.2 | 40 | 100% price, 47.5% rating |
| #4 | Reliance Digital | 37.5 | 29 | 0% price, 86.2% rating |

**Query Analysis:**
```
📈 TOP QUERIES (Most Products):
   1. "Samsung Galaxy S24" → 4 products
   2. "Dell Inspiron" → 4 products
   3. "HP Pavilion" → 4 products

📉 BOTTOM QUERIES (Least Products):
   1. "Apple HomePod" → 2 products
   2. "Gaming Laptop" → 3 products

🐢 SLOWEST QUERIES:
   1. "Mi TV" → 108.3s
   2. "Lenovo Tab" → 90.0s

⚡ FASTEST QUERIES:
   1. "Apple HomePod" → 49.4s
   2. "Sennheiser" → 50.3s
```

### 4.5 UMAP Clustering Validation

UMAP (Uniform Manifold Approximation and Projection) is used to visualize product clustering quality:

**Features Used for UMAP:**
- Text features (TF-IDF on product names): 50 dimensions
- Price feature: 1 dimension
- Rating feature: 1 dimension
- Sentiment score: 1 dimension
- **Total: 53 dimensions → 2D projection**

**Clustering Metrics:**
| Metric | Description | Single Agent | Multi-Agent |
|--------|-------------|--------------|-------------|
| **Silhouette Score** | -1 to 1, higher = better clustering | -0.29 | -0.19 |
| **Davies-Bouldin Index** | Lower = better separation | 7.23 | 9.00 |
| **Cluster Purity** | Category coherence % | 59.4% | 37.8% |

**Generated Visualizations:**
- `umap_single_agent_pkl.png` - Single agent clustering
- `umap_multi_agent_pkl.png` - Multi-agent clustering

### 4.6 Power Monitoring

The `PowerMonitor` class tracks resource consumption during scraping:

```python
class PowerMonitor:
    """Monitors CPU, memory, and estimates power consumption"""
    
    def generate_report(self):
        return {
            'resource_utilization': {
                'average_cpu_usage_percent': 15.1,
                'average_memory_usage_percent': 52.5
            },
            'power_consumption': {
                'average_cpu_power_watts': 2.8,
                'average_total_power_watts': 12.8
            },
            'energy_consumption': {
                'total_energy_kwh': 0.009474
            },
            'co2_emissions_grams': {
                'india': 7.769
            }
        }
```

**Power Comparison Results:**
| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Avg CPU Power | 0.00 W | 2.80 W |
| Avg Memory | 0.0% | 52.5% |
| Total Energy | 0.000 kWh | 0.009 kWh |
| CO₂ Emissions | 0.000 kg | 0.008 kg |

### 4.7 Test Reports Generated

| Report File | Contents |
|-------------|----------|
| `PKL_COMPARISON_REPORT.md` | Summary comparison tables |
| `PKL_DETAILED_ANALYSIS.md` | Full detailed analysis with all metrics |
| `COMPARISON_REPORT.md` | Single vs Multi-agent performance comparison |

---

## 5. Key Findings

### 5.1 Single Agent vs Multi-Agent Comparison

| Aspect | Single Agent | Multi-Agent | Winner |
|--------|--------------|-------------|--------|
| **Total Products** | 155 | 143 | Single Agent |
| **Data Accuracy** | 87.1% | 90.2% | Multi-Agent ✅ |
| **Source Coverage** | 3/4 | 4/4 | Multi-Agent ✅ |
| **Technical Details** | 41.3% | 68.5% | Multi-Agent ✅ |
| **Rating Data** | 48.4% | 81.1% | Multi-Agent ✅ |

### 5.2 Best Performing Websites

1. **Amazon.in**: Best for ratings and technical specifications
2. **Flipkart**: Best overall data quality with 97.1% rating completeness
3. **Croma**: Good product count, limited rating data
4. **Reliance Digital**: Price extraction challenges

### 5.3 Query Performance Insights

- **Best Category**: Laptops (15 products, 65.6s avg)
- **Challenging Category**: Smart TVs (14 products, 78.4s avg)
- **Fastest Queries**: Brand-specific searches (e.g., "Sennheiser")
- **Slowest Queries**: Generic searches (e.g., "Mi TV")

---

## Key Features Summary

| Feature | Technology | Benefit |
|---------|------------|---------|
| **Multi-Platform Scraping** | Selenium + 4 Parallel Threads | Amazon + Flipkart + Croma + Reliance Digital |
| **Smart Caching** | TF-IDF RAG | Instant results for repeated searches |
| **Neural Sentiment** | DistilBERT (91% accuracy) | Accurate sentiment from reviews |
| **Aspect-Based Analysis** | EnhancedSentimentAnalyzer | Quality, performance, battery insights |
| **Anti-Detection** | StealthBrowser + RateLimiter | Avoid bot detection and IP blocks |
| **Deep Review Scraping** | AdvancedReviewScraper | Up to 50 reviews with pagination |
| **Product Validation** | ProductValidator | Filter accessories, validate relevance |
| **UMAP Clustering** | UMAPAnalyzer | Visualize product similarity |
| **Power Monitoring** | PowerMonitor | Track energy consumption & CO₂ |
| **Comprehensive Testing** | comparison_test.py | 400-product validation suite |

### New Classes Added (January 2026)

| Class | Purpose |
|-------|---------|
| `AdvancedReviewScraper` | Deep review extraction with pagination support |
| `CromaScraper` | Scraper for Croma.com products |
| `RelianceDigitalScraper` | Scraper for RelianceDigital.in products |
| `StealthBrowser` | Anti-detection browser with stealth configuration |
| `RateLimiter` | Request throttling to avoid IP blocks |
| `EnhancedSentimentAnalyzer` | Multi-model sentiment with aspect-based analysis |
| `ProductValidator` | Brand/Series/Model validation with accessory filtering |
| `UMAPAnalyzer` | UMAP dimensionality reduction and clustering visualization |
| `PowerMonitor` | CPU/Memory/Energy consumption tracking |
| `UnifiedProductScraper` | Master coordinator for all platforms |

---

*Document updated for Major Project - January 2026*
*Neural Network Sentiment Analysis using DistilBERT*
*4-Platform Support: Amazon.in, Flipkart, Croma, Reliance Digital*
*Comprehensive Testing & Validation Suite*
