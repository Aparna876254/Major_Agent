# Chapter 5: Results and Analysis

This chapter presents a comprehensive evaluation of the e-commerce price comparison system, comparing single-agent sequential scraping with multi-agent parallel scraping across four major Indian platforms.

## 5.1 System Performance Evaluation

### 5.1.1 Dataset Overview

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| **Total Products Scraped** | 128 | 145 |
| **Total Categories Tested** | 10 | 10 |
| **Queries per Category** | 4 | 4 |
| **Total Search Queries** | 40 | 40 |
| **Single Agent Success Rate** | 100.0% | - |
| **Multi-Agent Success Rate** | - | 100.0% |

### 5.1.2 Product Distribution by Source

| E-Commerce Platform | Products Scraped | Percentage |
|---------------------|------------------|------------|
| **Amazon** | 39 | 26.9% |
| **Flipkart** | 37 | 25.5% |
| **Croma** | 40 | 27.6% |
| **Reliance Digital** | 29 | 20.0% |

### 5.1.3 Product Distribution by Category

| Product Category | Products Found |
|------------------|----------------|
| **Headphones** | 16 |
| **Gaming Consoles** | 16 |
| **Smartwatches** | 15 |
| **Tablets** | 15 |
| **Smart TVs** | 15 |
| **Cameras** | 15 |
| **Smart Speakers** | 15 |
| **Laptops** | 14 |
| **Wireless Earbuds** | 13 |
| **Smartphones** | 11 |

## 5.2 Time Efficiency Analysis

### 5.2.1 Overall Performance Metrics

| Performance Metric | Single Agent | Multi-Agent | Improvement |
|--------------------|--------------|-------------|-------------|
| **Total Execution Time** | 191.1 min | 47.3 min | +75.3% |
| **Average Time per Query** | 286.7s | 70.9s | - |
| **Throughput (Products/sec)** | 0.011 | 0.051 | +357.9% |
| **Products per Query** | 3.2 | 3.6 | - |

### 5.2.2 Latency Distribution Analysis

| Latency Metric | Value |
|----------------|-------|
| **Mean Latency** | 70.92s |
| **Median Latency** | 66.58s |
| **95th Percentile** | 103.21s |
| **99th Percentile** | 103.82s |
| **Standard Deviation** | 12.56s |
| **Min Latency** | 56.29s |
| **Max Latency** | 103.98s |

## 5.3 Bandwidth and Data Transfer Analysis

| Data Transfer Metric | Value |
|---------------------|-------|
| **Estimated Data per Product** | 15 KB |
| **Total Data Transferred** | 2.1 MB |
| **Average Bandwidth Usage** | 0.01 Mbps |
| **Data Efficiency** | 15.0 KB/product |

## 5.4 Resource Utilization & Energy Efficiency

| Resource Metric | Single Agent | Multi-Agent |
|-----------------|--------------|-------------|
| **Average CPU Usage** | 16.2% | 15.8% |
| **Average CPU Power** | 2.80W | 2.80W |
| **Memory Usage** | 71.0% | 55.2% |
| **Total Energy Consumption** | 0.040976 kWh | 0.010087 kWh |
| **CO₂ Emissions (India)** | 33.60g | 8.27g |

### 5.4.1 Energy Efficiency Analysis

| Efficiency Metric | Single Agent | Multi-Agent |
|-------------------|--------------|-------------|
| **Energy per Product** | 320.12 µWh | 69.57 µWh |
| **CO₂ per Product** | 0.2625g | 0.0570g |

## 5.5 Data Quality & Accuracy Assessment

### 5.5.1 Data Completeness Analysis

| Data Field | Complete Records | Completeness Rate |
|------------|------------------|-------------------|
| **Product Name** | 128/145 | 88.3% |
| **Price Information** | 116/145 | 80.0% |
| **Rating Data** | 72/145 | 49.7% |
| **Product Images** | 145/145 | 100.0% |
| **Sentiment Analysis** | 145/145 | 100.0% |

### 5.5.2 Price Distribution Analysis

| Price Metric | Value |
|--------------|-------|
| **Minimum Price** | ₹286 |
| **Maximum Price** | ₹224,470 |
| **Average Price** | ₹41,282 |
| **Median Price** | ₹29,494 |
| **Price Standard Deviation** | ₹42,577 |

## 5.6 Query Performance Analysis

### 5.6.1 Performance by Product Category

| Category | Avg Time/Query | Avg Products/Query | Total Products |
|----------|----------------|-------------------|----------------|
| **Smartphones** | 63.8s | 2.8 | 11 |
| **Headphones** | 65.5s | 4.0 | 16 |
| **Laptops** | 65.6s | 3.5 | 14 |
| **Gaming Consoles** | 66.0s | 4.0 | 16 |
| **Smartwatches** | 68.0s | 3.8 | 15 |
| **Cameras** | 69.6s | 3.8 | 15 |
| **Smart TVs** | 75.1s | 3.8 | 15 |
| **Smart Speakers** | 75.3s | 3.8 | 15 |
| **Wireless Earbuds** | 75.9s | 3.2 | 13 |
| **Tablets** | 84.3s | 3.8 | 15 |

## 5.7 UMAP Clustering Analysis

### 5.7.1 Clustering Methodology

The product clustering analysis was performed using **UMAP + K-Means**. 
Products were represented as TF-IDF vectors based on their names, descriptions, and features, 
then clustered to identify natural product groupings across different e-commerce platforms.

### 5.7.2 Clustering Quality Metrics

| Clustering Metric | Value | Interpretation |
|-------------------|-------|----------------|
| **Silhouette Score** | 0.139 | Cluster separation quality (-1 to 1, higher is better) |
| **Davies-Bouldin Index** | 2.12 | Cluster compactness (lower is better) |
| **Cluster Purity** | 57.9% | Category coherence within clusters |
| **Number of Clusters** | 10 | Detected product groups |

**Clustering Quality Assessment:** Moderate clustering quality - some overlap between clusters

### 5.7.3 Cluster-Category Mapping

| Cluster | Dominant Category | Purity | Size |
|---------|-------------------|--------|------|
| **Cluster 0** | Laptops | 76.5% | 17 products |
| **Cluster 1** | Gaming Consoles | 76.9% | 13 products |
| **Cluster 2** | Headphones | 50.0% | 22 products |
| **Cluster 3** | Smart TVs | 61.1% | 18 products |
| **Cluster 4** | Gaming Consoles | 16.7% | 24 products |
| **Cluster 5** | Cameras | 100.0% | 12 products |
| **Cluster 6** | Smartphones | 30.0% | 10 products |
| **Cluster 7** | Smart Speakers | 66.7% | 9 products |
| **Cluster 8** | Tablets | 83.3% | 12 products |
| **Cluster 9** | Smart Speakers | 50.0% | 8 products |

### 5.7.4 Cluster Size Distribution

| Cluster | Products | Percentage |
|---------|----------|------------|
| **Cluster 0** | 17 | 11.7% |
| **Cluster 1** | 13 | 9.0% |
| **Cluster 2** | 22 | 15.2% |
| **Cluster 3** | 18 | 12.4% |
| **Cluster 4** | 24 | 16.6% |
| **Cluster 5** | 12 | 8.3% |
| **Cluster 6** | 10 | 6.9% |
| **Cluster 7** | 9 | 6.2% |
| **Cluster 8** | 12 | 8.3% |
| **Cluster 9** | 8 | 5.5% |

## 5.8 Sentiment Analysis Validation

### 5.8.1 Sentiment Analysis Methodology

Sentiment analysis was performed using a **keyword-based approach**, analyzing product reviews 
and descriptions for positive and negative sentiment indicators.

### 5.8.2 Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Positive** | 68 | 46.9% |
| **Negative** | 2 | 1.4% |
| **Neutral** | 75 | 51.7% |

### 5.8.3 Sentiment Analysis Confidence

| Confidence Metric | Value |
|-------------------|-------|
| **Average Confidence** | 62.6% |
| **Median Confidence** | 55.0% |
| **Min Confidence** | 55.0% |
| **Max Confidence** | 85.0% |

### 5.8.4 Confidence Level Distribution

| Confidence Level | Count | Percentage |
|------------------|-------|------------|
| **High (≥80%)** | 10 | 6.9% |
| **Medium (60-79%)** | 60 | 41.4% |
| **Low (<60%)** | 75 | 51.7% |

### 5.8.5 Sentiment by Product Category

| Category | Positive | Neutral | Negative | Dominant Sentiment |
|----------|----------|---------|----------|-------------------|
| **Cameras** | 7 | 8 | 0 | Neutral |
| **Gaming Consoles** | 7 | 9 | 0 | Neutral |
| **Headphones** | 9 | 7 | 0 | Positive |
| **Laptops** | 3 | 11 | 0 | Neutral |
| **Smart Speakers** | 6 | 9 | 0 | Neutral |
| **Smart TVs** | 7 | 8 | 0 | Neutral |
| **Smartphones** | 6 | 4 | 1 | Positive |
| **Smartwatches** | 7 | 7 | 1 | Positive |
| **Tablets** | 8 | 7 | 0 | Positive |
| **Wireless Earbuds** | 8 | 5 | 0 | Positive |

## 5.9 Key Findings and Conclusions

1. **Performance Improvement**: Multi-agent approach achieved 75.3% faster execution time
2. **Throughput Enhancement**: 357.9% improvement in products per second
3. **Coverage Expansion**: Multi-agent scraped 17 additional products
4. **Source Diversity**: Successfully integrated 4/4 e-commerce platforms
5. **Reliability**: Achieved 100.0% success rate across all queries
6. **Product Clustering**: Identified 10 distinct product clusters with 57.9% category purity
7. **Sentiment Analysis**: 68/145 products (46.9%) have positive sentiment

---
*Report generated on 2026-01-01 07:06:47*