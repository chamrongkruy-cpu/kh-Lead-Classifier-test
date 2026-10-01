import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Lead Classifier - Cambodia",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔 Lead Classifier — Cambodia Nationwide")
st.markdown("Filter, classify, and deduplicate lead exports against the master database efficiently.")

# ---------------------------------------------------------
# DEFAULT CONFIGURATION & CATEGORY LIST
# ---------------------------------------------------------
DEFAULT_CATEGORIES = [
    # English & General F&B
    "restaurant", "cafe", "coffee", "bakery", "food", "noodle", "fast food", 
    "bubble tea", "bistro", "pub", "canteen", "eatery", "diner", "food court", 
    "food stall", "bar", "dessert", "beverage", "drink", "kitchen", "grill", 
    "khmer", "asian", "chinese", "japanese", "korean", "thai", "vietnamese", 
    "indian", "italian", "french", "western", "seafood", "bbq", "barbecue", 
    "hot pot", "sushi", "ramen", "pizza", "burger", "steak", "chicken", 
    "rice", "tea", "boba", "ice cream", "pastry", "donut", "sandwich", 
    "soup", "lok lak", "nom banh chok", "prahok", "kway teow", "deli", "cake",

    # Khmer Categories
    "ភោជនីយដ្ឋាន", "ហាង​កាហ្វេ", "ហាងលក់ភេសជ្ជៈ", "អាហារដ្ឋាន", 
    "កន្លែង​លក់​​អាហារ", "ផ្ទះបាយ", "អាហារ​ពេល​ល្ងាច", "ខេធើរីង", 
    "ហាង​លក់​ការ៉េម", "បា​តូច", "សេវា​ដឹកជញ្ជូន​ភីហ្សា", "ហាង​លក់​បង្អែម",

    # Chinese Categories
    "餐馆", "中餐馆", "烧烤", "火锅", "火锅店", "咖啡", "咖啡店", 
    "珍珠奶茶", "珍珠奶茶店", "糕餅店", "海鲜", "烧烤吧", "猫咪咖啡馆"
]

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("1. Category Settings")
fnb_input = st.sidebar.text_area(
    "Eligible F&B Categories (Comma Separated)", 
    ", ".join(DEFAULT_CATEGORIES),
    height=200
)
valid_categories = [c.strip().lower() for c in fnb_input.split(",") if c.strip()]

st.sidebar.header("2. Country Filter Settings")
country_filter = st.sidebar.text_input("Country Code Filter", "KH")

# ---------------------------------------------------------
# HELPER FUNCTIONS (OPTIMIZED FOR SPEED)
# ---------------------------------------------------------
def clean_phone(val):
    """Normalize phone numbers to last 8 digits for fast matching."""
    if pd.isna(val):
        return ""
    s = str(val).split('.')[0].strip()
    digits = ''.join(filter(str.isdigit, s))
    return digits[-8:] if len(digits) >= 8 else digits

# ---------------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------------
st.header("Upload Data Files")
col1, col2 = st.columns(2)

with col1:
    apify_file = st.file_uploader("Upload Scraped Leads (e.g. Apify Export CSV)", type=["csv"])

with col2:
    master_file = st.file_uploader("Upload Master Database (e.g. SF/Backend Accounts CSV)", type=["csv"])

# ---------------------------------------------------------
# PROCESSING PIPELINE
# ---------------------------------------------------------
if apify_file is not None:
    # 1. Load Leads Data
    df_leads = pd.read_csv(apify_file)
    st.info(f"📁 Loaded Input File: **{len(df_leads):,}** total rows.")

    # 2. Filter strictly for Cambodia (Nationwide, no city splitting)
    if 'countryCode' in df_leads.columns and country_filter:
        df_kh = df_leads[df_leads['countryCode'].astype(str).str.upper() == country_filter.upper()].copy()
        st.success(f"🇰🇭 Filtered for **{country_filter}** Nationwide: **{len(df_kh):,}** leads (Excluded {len(df_leads) - len(df_kh)} non-{country_filter} items).")
    else:
        df_kh = df_leads.copy()
        st.warning(f"⚠️ Column 'countryCode' not found. Processing all {len(df_kh):,} rows.")

    # 3. Category Eligibility Matching
    cat_cols = [c for c in df_kh.columns if c.startswith('categories/') or c in ['categoryName', 'categories']]
    
    if cat_cols:
        def check_fnb(row):
            combined_cats = " ".join([str(row[c]).lower() for c in cat_cols if pd.notnull(row[c])])
            return any(cat in combined_cats for cat in valid_categories)
        
        df_kh['Is_Eligible_F&B'] = df_kh.apply(check_fnb, axis=1)
    else:
        df_kh['Is_Eligible_F&B'] = True
        st.warning("⚠️ No category columns detected. Assuming all records are F&B eligible.")

    # 4. Master Database Lookup & Deduplication
    if master_file is not None:
        with st.spinner("⚡ Indexing Master Database for fast matching..."):
            df_master = pd.read_csv(master_file)
            
            # Create fast hash sets (O(1) lookups)
            existing_grids = set(df_master['GRID'].dropna().astype(str).str.strip().str.upper()) if 'GRID' in df_master.columns else set()
            
            if 'Phone' in df_master.columns:
                existing_phones = set(df_master['Phone'].apply(clean_phone).dropna())
                existing_phones.discard("")
            else:
                existing_phones = set()

            # Classify Lead Status
            def classify_lead(row):
                grid_val = str(row.get('GRID', '')).strip().upper()
                phone_val = clean_phone(row.get('phone', '')) if 'phone' in row else clean_phone(row.get('Phone', ''))
                
                if grid_val and grid_val in existing_grids:
                    return "Duplicate (GRID Match)"
                if phone_val and phone_val in existing_phones:
                    return "Duplicate (Phone Match)"
                return "New Lead"

            df_kh['Lead_Classification'] = df_kh.apply(classify_lead, axis=1)

    # ---------------------------------------------------------
    # DISPLAY & EXPORT RESULTS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Lead Classification Summary")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cambodia Leads", f"{len(df_kh):,}")
    c2.metric("Eligible F&B Leads", f"{df_kh['Is_Eligible_F&B'].sum():,}")
    
    if 'Lead_Classification' in df_kh.columns:
        new_count = (df_kh['Lead_Classification'] == "New Lead").sum()
        c3.metric("New Unclaimed Leads", f"{new_count:,}")

    # Results Table
    st.dataframe(
        df_kh[['title', 'countryCode', 'city', 'phone', 'categoryName', 'Is_Eligible_F&B'] + 
              (['Lead_Classification'] if 'Lead_Classification' in df_kh.columns else [])],
        use_container_width=True
    )

    # Download Cleaned Results
    csv_data = df_kh.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Classified Cambodia Leads CSV",
        data=csv_data,
        file_name="Classified_Cambodia_Leads.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Please upload your scraped leads CSV file above to begin.")
