# 🍔 Lead Classifier — Cambodia Nationwide

A high-performance **Streamlit** application designed to clean, classify, and deduplicate F&B (Food & Beverage) sales leads scraped from Google Maps or Apify against a master database (Salesforce / Backend Accounts).

---

## 🚀 Key Features

* **Nationwide Cambodia Scope (`KH`):** Filters leads exclusively for Cambodia without splitting or fragmenting data by individual cities.
* **Multi-Language Category Matching:** Pre-configured with F&B keywords in **English**, **Khmer** (e.g., ភោជនីយដ្ឋាន, ហាង​កាហ្វេ), and **Chinese** (e.g., 烧烤, 珍珠奶茶店, 火锅).
* **High-Speed Hash Matching ($O(1)$ Complexity):** Instantly deduplicates incoming leads against large master datasets (80,000+ records) using pre-indexed hash sets for GRID IDs and standardized 8-digit phone numbers.
* **Instant Export:** Download classified results directly as a clean CSV file.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/lead-classifier.git](https://github.com/your-username/lead-classifier.git)
   cd lead-classifier
