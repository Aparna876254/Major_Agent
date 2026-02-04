# 🛍️ E-Commerce Price Comparison Tool with RAG & Neural Sentiment Analysis

An intelligent web scraper that compares product prices across **4 major Indian e-commerce platforms** using Retrieval-Augmented Generation (RAG) for smart caching, semantic search, and **Neural Network-based Sentiment Analysis** powered by DistilBERT transformers.

## 🌟 Features

### Core Functionality
- **Multi-Platform Scraping**: Extracts product data from **Amazon.in, Flipkart, Croma, and Reliance Digital**
- **Deep Product Details**: Scrapes technical specifications, ratings, reviews, and descriptions
- **Real Customer Reviews Extraction**: Extracts actual customer comments (not just ratings) for accurate sentiment analysis
- **Smart Page Loading**: Intelligent scrolling and wait mechanisms to ensure all content loads properly
- **RAG-Based Caching**: Smart local database with semantic search capabilities
- **Intelligent Filtering**: Automatically filters accessories and validates product relevance
- **Anti-Detection Measures**: Stealth browser with human-like scrolling and rate limiting
- **Interactive GUI**: Rich interface displaying products with images, prices, sentiment indicators, and detailed specs

### 🧠 Neural Sentiment Analysis
- **Customer Review-Based**: Analyzes actual customer review text for authentic sentiment (not just star ratings)
- **Transformer-Based Model**: Uses DistilBERT fine-tuned on SST-2 from HuggingFace
- **High Accuracy**: ~91% accuracy with 66M parameter transformer model
- **Priority-Based Analysis**: Reviews → Description → Features (prioritizes real customer feedback)
- **Aspect-Based Analysis**: Analyzes specific aspects (quality, performance, battery, camera, display, value)
- **Visual Indicators**: Emoji-based sentiment display (😊 Positive, 😐 Neutral, 😞 Negative)
- **Confidence Scores**: Provides sentiment confidence percentages
- **Batch Processing**: Efficient analysis of multiple products simultaneously

### 🛡️ Advanced Scraping Features
- **StealthBrowser**: Anti-detection Chrome automation with randomized user agents
- **RateLimiter**: Request throttling to avoid IP blocks (10 requests/minute)
- **SmartRetryHandler**: Intelligent retry logic with exponential backoff
- **AdvancedReviewScraper**: Deep review extraction with pagination (up to 50 reviews)
- **Human-Like Scrolling**: Randomized scroll patterns to mimic real users
- **Smart Content Loading**: Extended wait times (5-8 seconds) and progressive scrolling to load lazy content
- **Multi-Selector Fallbacks**: Robust element detection with multiple CSS selectors per platform
- **Platform-Specific Selectors**: Tailored CSS selectors for Amazon, Flipkart, Croma, and Reliance Digital

### RAG Pipeline Strategy
1. **Local Exact Search** - Fast retrieval from cached products
2. **Fuzzy Matching** - Flexible search with 60% token matching
3. **External Scraping** - Fetches fresh data when cache misses
4. **Auto-Storage** - Grows knowledge base with each search

## 📋 Requirements

### Python Dependencies
```
selenium>=4.0.0
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
Pillow>=9.0.0
requests>=2.26.0
webdriver-manager>=3.8.0
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
matplotlib>=3.5.0
seaborn>=0.11.0
umap-learn>=0.5.0
psutil>=5.8.0
```

### System Requirements
- Python 3.8+
- Chrome Browser
- ChromeDriver (auto-installed via webdriver-manager)
- 8GB RAM minimum (recommended for neural models)
- Internet connection
- GPU optional (CUDA-enabled for faster inference)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Major Project"
```

2. **Install dependencies**
```bash
pip install selenium pandas scikit-learn numpy Pillow requests webdriver-manager transformers torch datasets matplotlib seaborn umap-learn psutil
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
# Single Agent Version
python Try.py

# Multi-Agent Version (with dedicated sentiment analysis agent)
python multi_agent_scraper.py
```

## 💻 Usage

### Main Menu Options

**1. Search Products**
- Enter product name (e.g., "samsung watch", "iphone 15")
- Specify number of products per source (default: 5)
- View results in interactive GUI with sentiment analysis

**2. View Database Statistics**
- Total products stored
- Price statistics (min, max, average, median)
- Rating statistics
- Source and category breakdown

**3. Clear Database**
- Remove all cached products
- Start fresh with new searches

**4. Exit**
- Close the application

### Search Examples
```
🔍 Enter product name: samsung galaxy watch
📦 Products per source: 5

🔍 Enter product name: iphone 15 pro
📦 Products per source: 3

🔍 Enter product name: oneplus nord
📦 Products per source: 10
```

## 🏗️ Architecture

### Project Structure
```
Major Project/
├── Try.py                          # Single-agent application (with advanced scrapers)
├── multi_agent_scraper.py          # Multi-agent application (4 platform support)
├── neural_sentiment_analyzer.py    # DistilBERT-based sentiment analysis
├── product_validator.py            # Product validation (Brand/Series/Model matching)
├── power_monitor.py                # CPU/Memory/Energy consumption monitoring
├── umap_rag_analyzer.py            # UMAP clustering and RAG storage visualization
├── comparison_test.py              # 400-product comparison test suite
├── analyze_pkl_results.py          # PKL file analysis with detailed reports
├── README.md                       # Documentation
├── PROJECT_DOCUMENTATION.md        # Detailed technical documentation
├── CHAPTER_5_RESULTS.md            # Chapter 5 results and analysis
├── CHAPTER5_GUIDE.md               # Guide for Chapter 5 report generation
├── requirements.txt                # Python dependencies
├── product_rag_database.pkl        # RAG storage (auto-generated)
└── __pycache__/                    # Python cache files
```

### Key Components

**1. RAG Pipeline** (`umap_rag_analyzer.py`)
- `RAGStorage`: Generic retrieval-augmented generation storage system
- `UMAPAnalyzer`: UMAP dimensionality reduction and clustering visualization
- `ProductRAGStorage`: Product-specific storage with TF-IDF vectorization
- Semantic search using cosine similarity

**2. Neural Sentiment Analyzer** (`neural_sentiment_analyzer.py`)
- `NeuralSentimentAnalyzer`: DistilBERT transformer model for sentiment analysis
- `EnhancedSentimentAnalyzer`: Multi-model analyzer with aspect-based analysis
- `DatasetLoader`: HuggingFace dataset loader for Amazon/Yelp reviews
- Pre-trained on SST-2 with ~91% accuracy

**3. Product Validator** (`product_validator.py`)
- `ProductValidator`: Brand/Series/Model validation with accessory filtering
- `ProductValidation`: Dataclass for validation results
- Handles all major smartphone and electronics brands

**4. Power Monitor** (`power_monitor.py`)
- `PowerMonitor`: CPU/Memory/Energy consumption tracking
- `Measurement`: Dataclass for power measurements
- CO₂ emissions calculation for different regions

**5. Web Scrapers** (`multi_agent_scraper.py`)
- `scrape_detailed_amazon()`: Amazon.in scraper with deep product details
- `scrape_detailed_flipkart()`: Flipkart scraper with retry logic
- `CromaScraper`: Croma.com product scraper
- `RelianceDigitalScraper`: RelianceDigital.in product scraper
- `AdvancedReviewScraper`: Deep review extraction with pagination (up to 50 reviews)
- `StealthBrowser`: Anti-detection browser automation
- `RateLimiter`: Request rate limiting to avoid blocks

**6. Multi-Agent System** (`multi_agent_scraper.py`)
- `AmazonAgent`: Dedicated agent for Amazon.in scraping
- `FlipkartAgent`: Dedicated agent for Flipkart scraping
- `CromaAgent`: Dedicated agent for Croma.com scraping
- `RelianceAgent`: Dedicated agent for RelianceDigital.in scraping
- `SentimentAgent`: Neural network sentiment analysis agent
- `FilterAgent`: Product filtering and validation agent
- `GUIAgent`: Results display agent

**7. Testing & Analysis**
- `comparison_test.py`: 400-product comprehensive test suite
- `analyze_pkl_results.py`: PKL analysis with Chapter 5 report generation

**5. Data Processing**
- `unified_rag_search()`: Orchestrates search workflow
- `filter_only_phones()`: Removes accessories for phone searches
- `categorize_product()`: Auto-categorizes products
- `clean_price()`: Normalizes price formats

**6. GUI**
- `display_results_gui_with_details()`: Interactive product comparison
- Async image loading
- Sentiment indicators with emoji and confidence scores
- Detailed product view with specifications and sentiment analysis
- Direct links to product pages

## 🎯 How It Works

### Search Workflow
```
User Query
    ↓
Local Exact Match (cached)
    ↓ (if not found)
Fuzzy Search (60% match)
    ↓ (if not found)
Web Scraping (Amazon + Flipkart + Croma + Reliance Digital)
    ↓
Validation & Filtering
    ↓
Neural Sentiment Analysis (DistilBERT)
    ↓
Store in RAG Database
    ↓
Display Results in GUI with Sentiment
```

### Sentiment Analysis Pipeline
```
Product Page Loaded
    ↓
Wait for Content (5s initial + scroll delays)
    ↓
Scroll to Reviews Section (8 steps × 0.8s)
    ↓
Extract Customer Review Text (not just star ratings!)
    ↓
Collect up to 5 Reviews with Actual Comments
    ↓
DistilBERT Tokenization (512 max tokens)
    ↓
Transformer Inference (GPU/CPU auto-detect)
    ↓
Aggregate Sentiment Scores from All Reviews
    ↓
Final: Sentiment Label + Confidence Score + Emoji
```

### Validation Logic
- **Accessory Detection**: Filters "back cover", "phone case", "screen protector", etc.
- **Brand Matching**: For brand-specific searches, validates brand presence
- **Generic Text Filtering**: Removes "Bestseller", "Coming Soon" placeholders
- **Price Validation**: Excludes products with invalid prices

### Scraping Strategy
- **Retry Logic**: 2 attempts with 30-second timeouts
- **Multiple Selectors**: Tries various CSS selectors for robustness
- **Tab Management**: Opens products in new tabs to preserve search results
- **Wait Times**: Random delays (3-10s) to avoid detection

## 📊 Data Extracted

### Basic Information
- Product name
- Price (numeric and formatted)
- Rating and review count
- Product image URL
- Source (Amazon/Flipkart)
- Product link
- Availability status

### Sentiment Information
- Sentiment label (Positive/Negative/Neutral)
- Confidence score (0-100%)
- Sentiment emoji (😊/😐/😞)
- Sentiment source (Customer Reviews / Product Description)
- Detailed explanation (e.g., "Based on 5 reviews: 4 positive, 1 neutral")

### Customer Reviews (for Sentiment Analysis)
- Actual review text (not just star ratings)
- Review title (e.g., "Fabulous!", "Awesome")
- Reviewer name (when available)
- Up to 5 reviews per product

### Detailed Information
- Technical specifications (dict)
- Additional product info (dict)
- Features and highlights (list)
- Full product description
- Category ratings (Flipkart: Camera, Battery, Display, Design)
- Rating breakdown (5★, 4★, 3★, 2★, 1★ counts)
- Category classification

## 🎯 Platform-Specific CSS Selectors

### Flipkart
| Data | CSS Selector |
|------|--------------|
| Rating | `span.PvbNMB`, `div._3LWZlK` |
| Review Text | `div.a6dZNm.mIW33x` |
| Specifications | `div.xdON2G`, `div.GNDEQ-` |
| Description | `div.KgDEGp`, `div.RmoJUa` |
| Highlights | `div._1mXcCf li` |
| Category Ratings | `div._2d4LTz` (Camera, Battery, Display, Design) |

### Amazon.in
| Data | CSS Selector |
|------|--------------|
| Price | `span.a-price-whole` |
| Reviews | `div.review-text-content` |
| Specs | `table.prodDetTable`, `div#productOverview_feature_div` |

### Croma
| Data | CSS Selector |
|------|--------------|
| Key Features | `div.key-features-box ul li` |
| Specifications | Accordion sections |
| Price | `span.pdp-price` |

### Reliance Digital
| Data | CSS Selector |
|------|--------------|
| Products | `.sp`, `.product-card` |
| URL Format | `/products?q={query}&page_no=1&page_size=12` |

## 🔧 Configuration

### Adjustable Parameters

**Search Settings**
```python
max_products = 5  # Products per source
target_count = max_products * 2  # Total target (both sources)
```

**Sentiment Analysis Settings**
```python
# Model configuration
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
max_length = 512  # Maximum tokens for analysis
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Validation Thresholds**
```python
fuzzy_match_threshold = 0.6  # 60% token match required
min_name_length = 10  # Minimum product name length
```

**Wait Times**
```python
initial_page_load = 5     # Seconds to wait for page load
scroll_pause = 0.8        # Seconds between scroll steps
scroll_steps = 8          # Number of scroll iterations
reviews_section_wait = 3  # Extra wait for reviews to load
page_load_timeout = 30    # Maximum wait for page
retry_attempts = 2        # Number of retries on failure
```

## 🛡️ Error Handling

- **Timeout Protection**: Graceful handling of slow-loading pages
- **Element Not Found**: Multiple selector fallbacks
- **Tab Management**: Auto-cleanup of browser tabs on errors
- **Data Validation**: Skips invalid products without crashing
- **Model Loading**: Graceful degradation if transformers unavailable

## 📈 Performance

### Optimization Features
- **Caching**: Reduces redundant scraping via RAG storage
- **Batch Processing**: Stores multiple products at once
- **Async Image Loading**: Non-blocking GUI image display
- **Vectorized Search**: Fast semantic similarity using TF-IDF
- **GPU Acceleration**: Optional CUDA support for neural models

### Neural Model Specifications
- **Model**: DistilBERT (66M parameters)
- **Layers**: 6 transformer layers
- **Max Tokens**: 512
- **Accuracy**: ~91% on SST-2 benchmark
- **Inference**: ~50ms per product (CPU), ~5ms (GPU)

### Typical Performance
- First search: 30-60 seconds (scraping + sentiment analysis)
- Cached search: <1 second (local retrieval)
- Sentiment analysis: ~50ms per product
- Products per search: 2-20 (configurable)
- Storage size: ~1MB per 100 products
- Model size: ~268MB (downloaded once)

---

## 🧪 Testing & Validation

### Test Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `comparison_test.py` | Compare single vs multi-agent (400 products) | `python comparison_test.py` |
| `analyze_pkl_results.py` | Analyze scraped data from .pkl files | `python analyze_pkl_results.py` |
| `product_validator.py` | Test product validation logic | Imported as module |

### Running Tests

```bash
# Run comprehensive comparison test (400 products, 10 categories)
python comparison_test.py

# Analyze existing pickle files
python analyze_pkl_results.py
```

### Product Validation

The `ProductValidator` ensures scraped products are relevant:

```python
from product_validator import ProductValidator

validator = ProductValidator()
is_valid, confidence, reason = validator.validate_product(
    "Apple iPhone 15 Pro 256GB", 
    "iPhone 15 Pro"
)
# Returns: (True, 0.95, "Brand and model match")
```

**Validation Rules:**
- ✅ Brand matching (Apple, Samsung, Sony, etc.)
- ✅ Series/Model matching (iPhone 15, Galaxy S24, etc.)
- ❌ Accessory filtering (cases, covers, screen protectors)
- ❌ Truncated name detection

### Performance Metrics

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| **Total Products** | 155 | 143 |
| **Data Accuracy** | 87.1% | 90.2% |
| **Source Coverage** | 3/4 | 4/4 |
| **Avg Latency** | N/A | 66.58s |
| **Throughput** | N/A | 0.054 prod/s |

### UMAP Clustering Analysis

Products are clustered using UMAP for quality validation:

```bash
# Generates visualization files:
# - umap_single_agent_pkl.png
# - umap_multi_agent_pkl.png
python analyze_pkl_results.py
```

**Clustering Metrics:**
| Metric | Description | Result |
|--------|-------------|--------|
| Silhouette Score | Cluster quality (-1 to 1) | -0.19 |
| Davies-Bouldin | Cluster separation (lower=better) | 9.00 |
| Cluster Purity | Category coherence | 37.8% |

### Power Monitoring

Track energy consumption during scraping:

```python
from power_monitor import PowerMonitor

monitor = PowerMonitor()
monitor.start_monitoring()

# ... scraping operations ...

report = monitor.generate_report()
print(f"Energy: {report['energy_consumption']['total_energy_kwh']} kWh")
print(f"CO₂: {report['co2_emissions_grams']['india']} grams")
```

### Generated Reports

After running tests, these reports are generated:

| File | Contents |
|------|----------|
| `PKL_COMPARISON_REPORT.md` | Summary comparison tables |
| `PKL_DETAILED_ANALYSIS.md` | Full latency/bandwidth/accuracy analysis |
| `COMPARISON_REPORT.md` | Single vs Multi-agent comparison |
| `umap_*.png` | UMAP clustering visualizations |

---

## ⚠️ Limitations

- **Website Changes**: Scrapers may break if Amazon/Flipkart update HTML structure
- **Rate Limiting**: Excessive requests may trigger anti-bot measures
- **Regional Availability**: Designed for Amazon.in and Flipkart India
- **Dynamic Content**: Some products may not load properly
- **No Authentication**: Cannot access user-specific prices or deals
- **Model Download**: First run requires ~268MB model download from HuggingFace

## 🔮 Future Enhancements

- [x] ~~Add more e-commerce platforms (Croma, Reliance Digital)~~ ✅ **DONE**
- [x] ~~Advanced review scraping with pagination~~ ✅ **DONE**
- [x] ~~Anti-detection measures (StealthBrowser)~~ ✅ **DONE**
- [x] ~~Aspect-based sentiment analysis~~ ✅ **DONE**
- [x] ~~Product validation with accessory filtering~~ ✅ **DONE**
- [x] ~~UMAP clustering visualization~~ ✅ **DONE**
- [x] ~~Power consumption monitoring~~ ✅ **DONE**
- [x] ~~Comprehensive test suite (400 products)~~ ✅ **DONE**
- [ ] Price history tracking and alerts
- [ ] Email notifications for price drops
- [ ] Export to Excel/CSV with charts
- [ ] Mobile app version
- [ ] API for third-party integration
- [ ] Fine-tune sentiment model on product-specific data
- [ ] Multi-language sentiment analysis support

## 📝 License

This project is for educational purposes. Respect the terms of service of Amazon and Flipkart when using this tool.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🙏 Acknowledgments

- Selenium WebDriver for browser automation
- ChromeDriver for Chrome integration
- scikit-learn for semantic search capabilities
- HuggingFace Transformers for neural sentiment analysis
- DistilBERT model from HuggingFace Hub
- UMAP-learn for dimensionality reduction
- psutil for power monitoring
- Flipkart scraping reference from [StackOverflow](https://stackoverflow.com/questions/28122882/) (CC BY-SA 3.0)

## 🧠 Neural Model Information

### Supported Datasets (for fine-tuning)
- **Amazon Polarity**: `mteb/amazon_polarity` - Binary sentiment classification
- **Amazon Reviews 2023**: `McAuley-Lab/Amazon-Reviews-2023` - Multi-class reviews
- **Yelp Reviews**: `Yelp/yelp_review_full` - 5-star rating prediction

### Pre-trained Model
- **DistilBERT SST-2**: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- Fine-tuned on Stanford Sentiment Treebank
- 66 million parameters
- 6 transformer layers

---

## 📊 Project Files

| File | Description |
|------|-------------|
| `Try.py` | Single-agent sequential scraper with RAG |
| `multi_agent_scraper.py` | Multi-agent parallel scraper (4 browser agents) |
| `neural_sentiment_analyzer.py` | DistilBERT sentiment analysis module |
| `product_validator.py` | Product validation with Brand/Series/Model matching |
| `power_monitor.py` | CPU/Memory/Energy consumption monitoring |
| `umap_rag_analyzer.py` | UMAP clustering, RAG storage, and visualization |
| `comparison_test.py` | 400-product comparison test suite |
| `analyze_pkl_results.py` | PKL file analysis with Chapter 5 reports |
| `CHAPTER_5_RESULTS.md` | Generated Chapter 5 results and analysis |
| `CHAPTER5_GUIDE.md` | Guide for Chapter 5 report generation |
| `PROJECT_DOCUMENTATION.md` | Complete technical documentation |
| `README.md` | This file |
| `requirements.txt` | Python dependencies |

---

**⚡ Built with Python | Powered by RAG & Neural Networks | Made for Smart Shopping**
**🧪 Validated with 400+ Products | 4 E-Commerce Platforms | Comprehensive Test Suite**
