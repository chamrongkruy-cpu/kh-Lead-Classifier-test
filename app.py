import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
import math
from typing import Tuple, Dict, Any, List, Optional

# Attempt optional library imports with safe fallbacks
try:
    from rapidfuzz import fuzz
except ImportError:
    st.error("Missing required package: rapidfuzz. Please run `pip install rapidfuzz`.")

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

try:
    from shapely.geometry import Point, Polygon
    from shapely.wkt import loads as load_wkt
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# Safe secret retrieval to prevent KeyError on Streamlit Cloud deployment
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None) if "APP_PASSWORD" in st.secrets else None


st.set_page_config(
    page_title="Sales Ops · Cambodia Lead Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.title("⚙️ Settings")

# Define available markets
MARKETS = {
    "Cambodia 🇰🇭 (KH)": "KH",
    "Singapore 🇸🇬 (SG)": "SG"
}

# Market selector dropdown with Cambodia pre-selected as default (index 0)
selected_market_label = st.sidebar.selectbox(
    "Market",
    options=list(MARKETS.keys()),
    index=0,  # Pre-selects Cambodia
    help="Select the region to load local matching rules and boundary settings."
)

market_code = MARKETS[selected_market_label]

st.sidebar.subheader("Match Thresholds")

p3_threshold = st.sidebar.slider(
    "P3 Potential Match starts at",
    min_value=0.30, max_value=0.80, value=0.50, step=0.05,
    help="Name similarity score range for manual rep verification."
)

p4_threshold = st.sidebar.slider(
    "P4 Duplicate starts at",
    min_value=0.60, max_value=0.95, value=0.75, step=0.05,
    help="Name similarity score at or above which a lead is marked as a Duplicate."
)

p3_threshold = st.sidebar.slider(
    "P3 Potential Match Threshold (%)",
    min_value=40, max_value=75, value=50, step=5,
    help="Name similarity score range for manual rep verification."
)

default_categories = [
    "restaurant", "cafe", "coffee", "bakery", "food", "noodle", "asian restaurant",
    "fast food", "bubble tea", "dessert", "barbecue", "khmer restaurant", "bistro"
]

fnb_categories_input = st.sidebar.text_area(
    "Eligible F&B Categories (comma separated)",
    value=", ".join(default_categories),
    help="Google Maps categories considered valid for food delivery onboarding."
)

valid_categories_list = [c.strip().lower() for c in fnb_categories_input.split(",") if c.strip()]


st.markdown('<p class="main-title">🎯 Sales Ops · Lead Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title"><b>Delivery Hero / foodpanda Cambodia</b> · Digital Sales APAC — Phnom Penh & Provinces</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Classify Leads",
    "🔗 Generate Apify URLs",
    "🏢 SF Account Audit",
    "📖 How to Use"
])


with tab1:
    st.subheader("1. Upload Input Files")
    st.caption("Upload the Salesforce Leads export, Apify Google Maps scrape results, and Salesforce All Accounts CRM export.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        leads_file = st.file_uploader("1️⃣ SF Leads File (.xlsx / .csv)", type=["xlsx", "csv"], key="leads")
    with col2:
        apify_file = st.file_uploader("2️⃣ Apify Results File (.xlsx / .csv)", type=["xlsx", "csv"], key="apify")
    with col3:
        crm_file = st.file_uploader("3️⃣ CRM All Accounts (.xlsx / .csv)", type=["xlsx", "csv"], key="crm")

    zones_file = st.file_uploader("🗺️ (Optional) Delivery Zones GeoJSON/JSON (`zones_KH.json`)", type=["json"], key="zones")

    # Load delivery zones if provided
    loaded_zones = []
    if zones_file:
        try:
            loaded_zones = json.load(zones_file)
            st.success(f"Loaded {len(loaded_zones)} delivery zones for coverage checking.")
        except Exception as e:
            st.error(f"Error loading zones JSON: {e}")


    if st.button("🚀 Run Lead Classification", type="primary", use_container_width=True):
        if not leads_file or not crm_file:
            st.error("Please upload at least the **Leads File** and the **CRM All Accounts File** to proceed.")
        else:
            with st.spinner("Processing files and executing Cambodian matching cascade..."):
                try:
                    # Read Files
                    df_leads = pd.read_excel(leads_file) if leads_file.name.endswith('.xlsx') else pd.read_csv(leads_file)
                    df_crm = pd.read_excel(crm_file) if crm_file.name.endswith('.xlsx') else pd.read_csv(crm_file)
                    
                    df_apify = None
                    if apify_file:
                        df_apify = pd.read_excel(apify_file) if apify_file.name.endswith('.xlsx') else pd.read_csv(apify_file)

                    # Resolve Columns for Leads
                    lead_cols = {
                        'grid': resolve_column(df_leads, ['GRID', 'Lead ID', 'Id', 'Lead_GRID']),
                        'name': resolve_column(df_leads, ['Company / Account', 'Company', 'Lead Name', 'Account Name', 'Name']),
                        'street': resolve_column(df_leads, ['Street / Street No.', 'Street', 'Address', 'Street Address']),
                        'sangkat': resolve_column(df_leads, ['Sangkat / Khan / Province', 'Sangkat', 'District', 'City', 'State']),
                        'lat': resolve_column(df_leads, ['Coordinates (Latitude)', 'Latitude', 'Lat', 'location/lat']),
                        'lng': resolve_column(df_leads, ['Coordinates (Longitude)', 'Longitude', 'Lng', 'Lng/Lat', 'location/lng'])
                    }

                    # Resolve Columns for CRM
                    crm_cols = {
                        'grid': resolve_column(df_crm, ['GRID', 'Account ID', 'Id']),
                        'name': resolve_column(df_crm, ['Account Name', 'Company Name', 'Name']),
                        'sangkat': resolve_column(df_crm, ['Sangkat / Khan', 'Sangkat', 'District', 'BillingCity']),
                        'lat': resolve_column(df_crm, ['Latitude', 'Lat', 'BillingLatitude']),
                        'lng': resolve_column(df_crm, ['Longitude', 'Lng', 'BillingLongitude']),
                        'address': resolve_column(df_crm, ['Formatted Restaurant Address', 'BillingStreet', 'Address'])
                    }

                    # Resolve Columns for Apify
                    apify_cols = {}
                    if df_apify is not None:
                        apify_cols = {
                            'grid': resolve_column(df_apify, ['GRID', 'lead_grid', 'Input_GRID']),
                            'title': resolve_column(df_apify, ['title', 'name', 'placeName']),
                            'category': resolve_column(df_apify, ['categoryName', 'category', 'primaryCategory']),
                            'perm_closed': resolve_column(df_apify, ['permanentlyClosed', 'permanently_closed']),
                            'temp_closed': resolve_column(df_apify, ['temporarilyClosed', 'temporarily_closed'])
                        }

                    results = []

                    # Iterate leads and apply rules
                    for idx, lead_row in df_leads.iterrows():
                        grid_val = lead_row.get(lead_cols.get('grid', ''), idx)
                        
                        # 1. CRM Proximity Match
                        crm_label, score, match_acc, match_reason = find_crm_matches(
                            lead_row, df_crm, lead_cols, crm_cols,
                            radius_meters=proximity_radius,
                            p4_threshold=p4_threshold,
                            p3_threshold=p3_threshold
                        )

                        final_label = crm_label
                        reason = match_reason
                        matched_crm_name = match_acc.get(crm_cols.get('name', ''), '') if match_acc is not None else ""
                        matched_crm_grid = match_acc.get(crm_cols.get('grid', ''), '') if match_acc is not None else ""

                        # 2. Apify Google Maps validation if no CRM match
                        if crm_label == "No CRM Match":
                            apify_match_row = None
                            if df_apify is not None and apify_cols.get('grid'):
                                matched_rows = df_apify[df_apify[apify_cols['grid']].astype(str) == str(grid_val)]
                                if not matched_rows.empty:
                                    apify_match_row = matched_rows.iloc[0]

                            apify_label, apify_reason = validate_apify_status(
                                apify_match_row, apify_cols, valid_categories_list
                            )
                            final_label = apify_label
                            reason = apify_reason

                        # 3. Zone checking
                        in_zone = True
                        zone_name = "N/A"
                        if loaded_zones and lead_cols.get('lat') and lead_cols.get('lng'):
                            lat_v = lead_row.get(lead_cols['lat'])
                            lng_v = lead_row.get(lead_cols['lng'])
                            in_zone, zone_name = check_delivery_zone(lat_v, lng_v, loaded_zones)

                        # Output Row Construction
                        res_row = lead_row.to_dict()
                        res_row['Final Classification'] = final_label
                        res_row['Classification Reason'] = reason
                        res_row['Match Score (%)'] = round(score, 1)
                        res_row['Matched CRM Account Name'] = matched_crm_name
                        res_row['Matched CRM GRID'] = matched_crm_grid
                        res_row['Delivery Zone Coverage'] = zone_name if loaded_zones else "Not Checked"
                        
                        results.append(res_row)

                    out_df = pd.DataFrame(results)

                    st.divider()
                    st.subheader("2. Classification Results Summary")

                    # Metric cards
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Leads", len(out_df))
                    m2.metric("✅ P1 — New", len(out_df[out_df['Final Classification'] == "P1 — New"]))
                    m3.metric("🔴 P4 — Duplicate", len(out_df[out_df['Final Classification'] == "P4 — Duplicate"]))
                    m4.metric("🟡 P3 — Potential", len(out_df[out_df['Final Classification'] == "P3 — Potential Match"]))
                    m5.metric("⚪ P2 / Closed", len(out_df[out_df['Final Classification'].isin(["P2 — Please Check", "Business Closed", "Wrong Target Group"])))

                    st.dataframe(out_df, use_container_width=True)

                    # Export button
                    excel_bytes = export_to_excel(out_df)
                    st.download_button(
                        label="📥 Download Excel Report (6 Color-Coded Sheets)",
                        data=excel_bytes,
                        file_name="Cambodia_Classified_Leads_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )

                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
                    st.exception(e)


with tab2:
    st.subheader("🔗 Generate Apify Google Maps Search URLs")
    st.caption("Generate targeted Google Maps search URLs for Cambodian cities and provinces to feed into the Apify scraper.")

    raw_leads_text = st.text_area(
        "Paste Restaurant Names & Sangkats (one per line)",
        value="Bay Cha BKK1, Phnom Penh\nNum Banh Chok Toul Kork, Phnom Penh\nPub Street Cafe, Siem Reap",
        height=150
    )

    if st.button("Generate Search URLs"):
        lines = [line.strip() for line in raw_leads_text.split('\n') if line.strip()]
        url_data = []
        for line in lines:
            query = f"{line}, Cambodia"
            encoded_q = re.sub(r'\s+', '+', query)
            url = f"https://www.google.com/maps/search/{encoded_q}"
            url_data.append({"Search Query": line, "Google Maps Search URL": url})
        
        url_df = pd.DataFrame(url_data)
        st.dataframe(url_df, use_container_width=True)


with tab3:
    st.subheader("🏢 Salesforce CRM Internal Duplicate Audit")
    st.caption("Upload your Salesforce CRM All Accounts list to discover duplicate records already existing inside Salesforce.")

    crm_audit_file = st.file_uploader("Upload CRM Accounts File for Internal Audit", type=["xlsx", "csv"], key="crm_audit")

    if crm_audit_file and st.button("Run Internal Audit"):
        with st.spinner("Analyzing CRM records for duplicates..."):
            df_audit = pd.read_excel(crm_audit_file) if crm_audit_file.name.endswith('.xlsx') else pd.read_csv(crm_audit_file)
            name_col = resolve_column(df_audit, ['Account Name', 'Name', 'Company'])
            lat_col = resolve_column(df_audit, ['Latitude', 'Lat'])
            lng_col = resolve_column(df_audit, ['Longitude', 'Lng'])

            if not name_col:
                st.error("Could not locate 'Account Name' column in uploaded file.")
            else:
                duplicates = []
                num_records = min(len(df_audit), 500) # Cap for quick performance demonstration
                records = df_audit.head(num_records).to_dict('records')

                for i in range(len(records)):
                    for j in range(i + 1, len(records)):
                        r1, r2 = records[i], records[j]
                        n1, n2 = normalize_cambodian_text(r1.get(name_col)), normalize_cambodian_text(r2.get(name_col))
                        
                        score = fuzz.token_set_ratio(n1, n2)
                        if score >= p4_threshold:
                            duplicates.append({
                                "Account 1": r1.get(name_col),
                                "Account 2": r2.get(name_col),
                                "Similarity Score (%)": score,
                                "Audit Action": "Flagged Internal Duplicate"
                            })

                if duplicates:
                    st.warning(f"Found {len(duplicates)} duplicate pairs within Salesforce CRM records!")
                    st.dataframe(pd.DataFrame(duplicates), use_container_width=True)
                else:
                    st.success("No internal duplicate records found above threshold.")


with tab4:
    st.markdown("""
    ### 📖 Cambodian Lead Classifier Guide
    
    #### Matching Logic Overview
    1. **Geographic Proximity First:** Evaluates CRM accounts within a set radius (default 200m) or within the same **Sangkat / Khan**.
    2. **Multilingual Text Matching:** Normalizes Khmer script (`ភាសាខ្មែរ`), Romanized Khmer transliterations, and English characters.
    3. **Apify Google Maps Validation:** Verifies if non-CRM matched leads are active open food venues.

    #### Required Salesforce Export Fields
    - `Company / Account`
    - `Street / Street No.`
    - `Sangkat / Khan / Province`
    - `Coordinates (Latitude)` & `Coordinates (Longitude)`
    """)
