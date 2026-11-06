import io
import json
import math
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Attempt optional library imports with safe fallbacks
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)

st.set_page_config(
    page_title="Sales Ops · Cambodia Lead Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for foodpanda / Delivery Hero Cambodia branding
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #D70F64;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A4A4A;
        margin-bottom: 1.5rem;
    }
    .stButton>button[kind="primary"] {
        background-color: #D70F64 !important;
        border-color: #D70F64 !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #B50B52 !important;
        border-color: #B50B52 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)


def normalize_cambodian_text(text: Any) -> str:
    if pd.isna(text) or text is None:
        return ""
    s = str(text).strip().lower()
    if not s:
        return ""

    noise_patterns = [
        r"\bco\.?,?\s*ltd\.?\b",
        r"\bco\.?,?\s*ltd\b",
        r"\binc\.?\b",
        r"\bexpress\b",
        r"\bcambodia\b",
        r"\bphnom\s*penh\b",
        r"\bkhmer\b",
        r"\benterprise\b",
        r"\bgroup\b",
    ]
    for pattern in noise_patterns:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)

    s = re.sub(r"[^\w\s\u1780-\u17ff]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(
        phi2
    ) * math.sin(delta_lambda / 2.0) ** 2
    return R * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def calculate_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    try:
        if HAS_GEOPY:
            return geodesic((lat1, lon1), (lat2, lon2)).meters
        return haversine_distance(lat1, lon1, lat2, lon2)
    except Exception:
        return float("inf")


def resolve_column(
    df: pd.DataFrame, possible_names: List[str]
) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols_lower = {str(c).lower().strip(): str(c) for c in df.columns}
    for name in possible_names:
        nl = name.lower().strip()
        if nl in cols_lower:
            return cols_lower[nl]
    return None


def extract_apify_coords(
    row: pd.Series, apify_cols: Dict[str, str]
) -> Tuple[Optional[float], Optional[float]]:
    """Safely extracts latitude and longitude from flat or nested JSON Apify columns."""
    lat_col = apify_cols.get("lat")
    lng_col = apify_cols.get("lng")

    if lat_col and lng_col and pd.notnull(row.get(lat_col)) and pd.notnull(row.get(lng_col)):
        try:
            return float(row[lat_col]), float(row[lng_col])
        except (ValueError, TypeError):
            pass

    loc_col = apify_cols.get("location_json")
    if loc_col and pd.notnull(row.get(loc_col)):
        raw_val = row[loc_col]
        if isinstance(raw_val, dict):
            try:
                return float(raw_val.get("lat")), float(raw_val.get("lng"))
            except (ValueError, TypeError):
                pass
        elif isinstance(raw_val, str) and "{" in raw_val:
            try:
                clean_json = raw_val.replace("'", '"')
                parsed = json.loads(clean_json)
                return float(parsed.get("lat")), float(parsed.get("lng"))
            except Exception:
                pass

    return None, None


def match_lead_to_apify(
    lead_name: str,
    lead_sangkat: str,
    lead_lat: Optional[float],
    lead_lng: Optional[float],
    df_apify: pd.DataFrame,
    apify_cols: Dict[str, str],
    max_radius_meters: float = 350.0,
) -> Tuple[Optional[pd.Series], str]:
    """Matches a lead to Apify without GRID, enforcing geographic distance if coords exist."""
    if df_apify is None or df_apify.empty or not lead_name:
        return None, "No Apify data available"

    norm_lead = normalize_cambodian_text(lead_name)
    if not norm_lead:
        return None, "Empty lead name after normalization"

    has_lead_coords = (
        lead_lat is not None
        and lead_lng is not None
        and not math.isnan(lead_lat)
        and not math.isnan(lead_lng)
    )

    candidates = []

    # Strategy 1: URL / Search Query parameter text matching
    url_col = apify_cols.get("input_url")
    if url_col and url_col in df_apify.columns:
        url_matches = df_apify[
            df_apify[url_col]
            .astype(str)
            .str.lower()
            .str.contains(re.escape(norm_lead), na=False)
        ]
        if not url_matches.empty:
            candidates.extend(url_matches.to_dict("records"))

    # Strategy 2: Place Title / Business Name fuzzy matching
    title_col = apify_cols.get("title")
    if title_col and title_col in df_apify.columns:
        for _, row in df_apify.iterrows():
            cand_name = normalize_cambodian_text(row.get(title_col, ""))
            if not cand_name:
                continue

            if HAS_RAPIDFUZZ:
                score = (
                    max(
                        fuzz.token_set_ratio(norm_lead, cand_name),
                        fuzz.token_sort_ratio(norm_lead, cand_name),
                    )
                    / 100.0
                )
            else:
                score = 1.0 if norm_lead == cand_name else (0.6 if norm_lead in cand_name else 0.0)

            if score >= 0.75:
                row_dict = row.to_dict()
                row_dict["_match_score"] = score
                candidates.append(row_dict)

    if not candidates:
        return None, "No Apify name/query match"

    df_cand = pd.DataFrame(candidates)
    if title_col and title_col in df_cand.columns:
        df_cand = df_cand.drop_duplicates(subset=[title_col])

    # Distance Gating
    for _, cand_row in df_cand.iterrows():
        cand_lat, cand_lng = extract_apify_coords(cand_row, apify_cols)

        if has_lead_coords and cand_lat is not None and cand_lng is not None:
            dist_m = calculate_distance(lead_lat, lead_lng, cand_lat, cand_lng)
            if dist_m <= max_radius_meters:
                return cand_row, f"Matched within {int(dist_m)}m"
            else:
                continue
        else:
            return cand_row, "Matched (Without coordinate validation)"

    return None, f"Found matching name, but outside {int(max_radius_meters)}m radius"


def find_crm_matches(
    lead_row: pd.Series,
    crm_df: pd.DataFrame,
    lead_cols: Dict[str, str],
    crm_cols: Dict[str, str],
    radius_meters: float = 200.0,
    p4_threshold: float = 0.75,
    p3_threshold: float = 0.50,
) -> Tuple[str, float, Optional[pd.Series], str]:
    lead_name = normalize_cambodian_text(lead_row.get(lead_cols.get("name", "")))
    if not lead_name:
        return "P2 — Please Check", 0.0, None, "No lead name provided or normalized to empty"

    lead_lat = lead_row.get(lead_cols.get("lat", ""))
    lead_lng = lead_row.get(lead_cols.get("lng", ""))
    lead_sangkat = normalize_cambodian_text(lead_row.get(lead_cols.get("sangkat", "")))

    has_coords = pd.notnull(lead_lat) and pd.notnull(lead_lng)
    try:
        if has_coords:
            lead_lat = float(lead_lat)
            lead_lng = float(lead_lng)
    except Exception:
        has_coords = False

    best_score = 0.0
    best_match_row = None
    match_reason = "No CRM Match"

    # Pre-filtering candidates using Sangkat if available
    candidate_crm = crm_df
    if lead_sangkat and crm_cols.get("sangkat") and "_norm_sangkat" in crm_df.columns:
        sangkat_mask = (
            crm_df["_norm_sangkat"].str.contains(lead_sangkat, regex=False)
            | crm_df["_norm_sangkat"].apply(lambda x: x in lead_sangkat if x else False)
        )
        if sangkat_mask.any():
            candidate_crm = crm_df[sangkat_mask]

    for _, crm_row in candidate_crm.iterrows():
        crm_lat = crm_row.get(crm_cols.get("lat", ""))
        crm_lng = crm_row.get(crm_cols.get("lng", ""))
        crm_sangkat = crm_row.get("_norm_sangkat", "")

        in_proximity = False
        dist_m = float("inf")

        if has_coords and pd.notnull(crm_lat) and pd.notnull(crm_lng):
            try:
                dist_m = calculate_distance(lead_lat, lead_lng, float(crm_lat), float(crm_lng))
                if dist_m <= radius_meters:
                    in_proximity = True
            except Exception:
                pass

        if not in_proximity and lead_sangkat and crm_sangkat:
            if lead_sangkat in crm_sangkat or crm_sangkat in lead_sangkat:
                in_proximity = True

        if in_proximity:
            crm_name = crm_row.get("_norm_name", "")
            if not crm_name:
                continue

            if HAS_RAPIDFUZZ:
                score = (
                    max(
                        fuzz.token_set_ratio(lead_name, crm_name),
                        fuzz.token_sort_ratio(lead_name, crm_name),
                    )
                    / 100.0
                )
            else:
                score = 1.0 if lead_name == crm_name else (0.6 if lead_name in crm_name else 0.0)

            if score > best_score:
                best_score = score
                best_match_row = crm_row
                match_reason = (
                    f"Proximity ({int(dist_m)}m)"
                    if dist_m < float("inf")
                    else f"Sangkat Match ({crm_sangkat})"
                )

    if best_score >= p4_threshold:
        return (
            "P4 — Duplicate",
            best_score * 100.0,
            best_match_row,
            f"Name ≥ {int(p4_threshold*100)}% at same location ({match_reason})",
        )
    elif best_score >= p3_threshold:
        return (
            "P3 — Potential Match",
            best_score * 100.0,
            best_match_row,
            f"Name {int(p3_threshold*100)}–{int(p4_threshold*100)-1}% at same location ({match_reason})",
        )
    else:
        return "No CRM Match", 0.0, None, "No CRM match above threshold"


def validate_apify_status(
    apify_row: Optional[pd.Series],
    apify_cols: Dict[str, str],
    valid_categories: List[str],
) -> Tuple[str, str]:
    if apify_row is None:
        return "P2 — Please Check", "No Apify result found"

    cat_val = str(apify_row.get(apify_cols.get("category", ""), "")).lower().strip()
    perm_closed = str(apify_row.get(apify_cols.get("perm_closed", ""), "")).lower() in ["true", "1", "yes"]
    temp_closed = str(apify_row.get(apify_cols.get("temp_closed", ""), "")).lower() in ["true", "1", "yes"]

    if perm_closed or temp_closed:
        return "Business Closed", "Apify: Google confirms permanently/temporarily closed"

    if not cat_val:
        return "P2 — Please Check", "No Apify category found"

    is_eligible = any(cat in cat_val or cat_val in cat for cat in valid_categories)
    if is_eligible:
        return "P1 — New", "No CRM match, Apify-confirmed restaurant"
    else:
        return "Wrong Target Group", f"Apify: category not food-delivery eligible ({cat_val})"


def export_to_excel(df_results: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    if not HAS_OPENPYXL:
        df_results.to_csv(output, index=False)
        return output.getvalue()

    wb = Workbook()
    fill_p1 = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    fill_p4 = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    fill_p3 = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    fill_p2 = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
    header_fill = PatternFill(start_color="D70F64", end_color="D70F64", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

    ws_all = wb.active
    ws_all.title = "Classified Leads"
    headers = list(df_results.columns)
    ws_all.append(headers)

    for col_num in range(1, len(headers) + 1):
        cell = ws_all.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font

    class_col_idx = headers.index("Final Classification") + 1

    for _, row in df_results.iterrows():
        ws_all.append(list(row))
        curr_row = ws_all.max_row
        lbl = str(row.get("Final Classification", ""))
        target_cell = ws_all.cell(row=curr_row, column=class_col_idx)

        if "P1" in lbl:
            target_cell.fill = fill_p1
        elif "P4" in lbl:
            target_cell.fill = fill_p4
        elif "P3" in lbl:
            target_cell.fill = fill_p3
        elif "P2" in lbl or "Closed" in lbl or "Wrong" in lbl:
            target_cell.fill = fill_p2

    sheets_config = [
        ("✅ P1 — New", df_results[df_results["Final Classification"] == "P1 — New"]),
        ("🔴 P4 — Duplicate", df_results[df_results["Final Classification"] == "P4 — Duplicate"]),
        ("🟡 P3 — Potential", df_results[df_results["Final Classification"] == "P3 — Potential Match"]),
        ("⚪ P2 — Please Check", df_results[df_results["Final Classification"] == "P2 — Please Check"]),
        (
            "⚠️ Closed & Wrong TG",
            df_results[
                df_results["Final Classification"].isin(
                    ["Business Closed", "Wrong Target Group"]
                )
            ],
        ),
    ]

    for title, sub_df in sheets_config:
        ws = wb.create_sheet(title=title)
        ws.append(headers)
        for col_num in range(1, len(headers) + 1):
            c = ws.cell(row=1, column=col_num)
            c.fill = header_fill
            c.font = header_font
        for _, r in sub_df.iterrows():
            ws.append(list(r))

    wb.save(output)
    return output.getvalue()


# Sidebar Configuration
st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("🎯 Matching Thresholds")
p3_threshold = st.sidebar.slider("P3 Potential Match starts at (%)", 40, 80, 50, 5) / 100.0
p4_threshold = st.sidebar.slider("P4 Duplicate starts at (%)", 60, 95, 75, 5) / 100.0

st.sidebar.subheader("📍 Geospatial Radii")
crm_proximity_radius = st.sidebar.slider("CRM Duplicate Radius (Meters)", 50, 1000, 200, 50)
apify_proximity_radius = st.sidebar.slider("Apify Place Validation Radius (Meters)", 50, 1500, 350, 50)

# Expanded Default Categories for Cambodia F&B
default_categories = [
    "restaurant", "cafe", "coffee", "bakery", "food", "noodle", "fast food",
    "bubble tea", "bistro", "pub", "canteen", "eatery", "diner", "food court",
    "food stall", "bar", "dessert", "beverage", "drink", "kitchen", "grill",
    "khmer", "asian", "chinese", "japanese", "korean", "thai", "vietnamese",
    "indian", "italian", "french", "western", "seafood", "bbq", "barbecue",
    "hot pot", "sushi", "ramen", "pizza", "burger", "steak", "chicken",
    "rice", "tea", "boba", "ice cream", "pastry", "donut", "sandwich",
    "soup", "lok lak", "nom banh chok", "prahok", "kway teow"
]

st.sidebar.subheader("🍽️ Target Group Filtering")
fnb_categories_input = st.sidebar.text_area(
    "Eligible F&B Categories",
    value=", ".join(default_categories),
    height=140,
    help="Comma-separated list of accepted Google Maps categories.",
)
valid_categories_list = [
    c.strip().lower()
    for c in re.split(r"[,\n]", fnb_categories_input)
    if c.strip()
]

st.markdown(
    '<p class="main-title">🎯 Sales Ops · Lead Classifier</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title"><b>Delivery Hero / foodpanda Cambodia</b> · Digital Sales APAC — Phnom Penh & Provinces</p>',
    unsafe_allow_html=True,
)

if APP_PASSWORD:
    pwd_input = st.sidebar.text_input("🔑 App Password", type="password")
    if pwd_input != APP_PASSWORD:
        st.info("🔒 Please enter the correct password in the sidebar to access the classifier tool.")
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Classify Leads",
    "🔗 Generate Apify URLs & Preview",
    "🏢 SF Account Audit",
    "📖 How to Use",
])

with tab1:
    with st.expander("📎 How to get your files — click to expand", expanded=False):
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("**Step 1 · Leads file (Salesforce)**")
            st.link_button(
                "Open Leads Report →",
                "https://deliveryhero.lightning.force.com/lightning/r/Report/00ObO000008clm5UAA/view?",
                use_container_width=True,
            )
        with exp_col2:
            st.markdown("**Step 2 · CRM Export (Salesforce)**")
            st.link_button(
                "Open Cambodia CRM Report →",
                "https://deliveryhero.lightning.force.com/lightning/r/Report/00ObO000008cmAHUAY/view?queryScope=userFolders",
                use_container_width=True,
            )
        st.info(
            "🔗 **Note:** Classification matches leads to Apify places using **Name, Search Query, and Lat/Lng Proximity** without requiring matching GRID codes."
        )

    st.subheader("1. Upload Input Files")
    col1, col2, col3 = st.columns(3)

    with col1:
        leads_file = st.file_uploader(
            "1️⃣ SF Leads File (.xlsx / .csv)",
            type=["xlsx", "csv"],
            key="leads",
        )
    with col2:
        apify_file = st.file_uploader(
            "2️⃣ Apify Results File (.xlsx / .csv)",
            type=["xlsx", "csv"],
            key="apify",
        )
        if "loaded_apify_df" in st.session_state:
            st.success("✅ Apify results auto-loaded from Tab 2!")
    with col3:
        crm_file = st.file_uploader(
            "3️⃣ CRM All Accounts (.xlsx / .csv)",
            type=["xlsx", "csv"],
            key="crm",
        )

    if st.button("🚀 Run Lead Classification", type="primary", use_container_width=True):
        if not leads_file or not crm_file:
            st.error("Please upload at least the **Leads File** and the **CRM All Accounts File** to proceed.")
        else:
            with st.spinner("Classifying leads..."):
                try:
                    df_leads = (
                        pd.read_excel(leads_file)
                        if leads_file.name.endswith(".xlsx")
                        else pd.read_csv(leads_file)
                    )
                    df_crm = (
                        pd.read_excel(crm_file)
                        if crm_file.name.endswith(".xlsx")
                        else pd.read_csv(crm_file)
                    )

                    df_apify = None
                    if "loaded_apify_df" in st.session_state:
                        df_apify = st.session_state["loaded_apify_df"]
                    elif apify_file:
                        df_apify = (
                            pd.read_excel(apify_file)
                            if apify_file.name.endswith(".xlsx")
                            else pd.read_csv(apify_file)
                        )

                    lead_cols = {
                        "name": resolve_column(
                            df_leads, ["Company / Account", "Company", "Lead Name", "Name"]
                        ),
                        "street": resolve_column(
                            df_leads, ["Street / Street No.", "Street", "Address"]
                        ),
                        "sangkat": resolve_column(
                            df_leads, ["Sangkat / Khan / Province", "Sangkat", "District", "City"]
                        ),
                        "lat": resolve_column(
                            df_leads, ["Coordinates (Latitude)", "Latitude", "Lat"]
                        ),
                        "lng": resolve_column(
                            df_leads, ["Coordinates (Longitude)", "Longitude", "Lng"]
                        ),
                        "grid": resolve_column(df_leads, ["GRID", "Lead ID", "Id"]),
                    }

                    crm_cols = {
                        "name": resolve_column(df_crm, ["Account Name", "Name"]),
                        "sangkat": resolve_column(df_crm, ["Sangkat / Khan", "Sangkat", "District"]),
                        "lat": resolve_column(df_crm, ["Latitude", "Lat"]),
                        "lng": resolve_column(df_crm, ["Longitude", "Lng"]),
                        "grid": resolve_column(df_crm, ["GRID", "Account ID", "Id"]),
                    }

                    # Pre-calculate normalized CRM values for faster vector checks
                    if crm_cols.get("name"):
                        df_crm["_norm_name"] = df_crm[crm_cols["name"]].apply(normalize_cambodian_text)
                    else:
                        df_crm["_norm_name"] = ""

                    if crm_cols.get("sangkat"):
                        df_crm["_norm_sangkat"] = df_crm[crm_cols["sangkat"]].apply(normalize_cambodian_text)
                    else:
                        df_crm["_norm_sangkat"] = ""

                    apify_cols = {}
                    if df_apify is not None:
                        apify_cols = {
                            "title": resolve_column(df_apify, ["title", "name", "Place Name", "place_name"]),
                            "input_url": resolve_column(
                                df_apify,
                                [
                                    "inputStartUrl", "inputUrl", "searchUrl", "url",
                                    "input_url", "startUrl", "query", "url/url",
                                    "input/url", "Search Query",
                                ],
                            ),
                            "category": resolve_column(df_apify, ["categoryName", "category"]),
                            "perm_closed": resolve_column(df_apify, ["permanentlyClosed", "permanently_closed"]),
                            "temp_closed": resolve_column(df_apify, ["temporarilyClosed", "temporarily_closed"]),
                            "lat": resolve_column(df_apify, ["location/lat", "latitude", "lat", "location.lat"]),
                            "lng": resolve_column(df_apify, ["location/lng", "longitude", "lng", "location.lng"]),
                            "location_json": resolve_column(df_apify, ["location"]),
                        }

                    results = []
                    for idx, lead_row in df_leads.iterrows():
                        crm_label, score, match_acc, match_reason = find_crm_matches(
                            lead_row,
                            df_crm,
                            lead_cols,
                            crm_cols,
                            radius_meters=crm_proximity_radius,
                            p4_threshold=p4_threshold,
                            p3_threshold=p3_threshold,
                        )

                        final_label = crm_label
                        reason = match_reason
                        matched_crm_name = (
                            match_acc.get(crm_cols.get("name", ""), "")
                            if match_acc is not None
                            else ""
                        )
                        matched_crm_grid = (
                            match_acc.get(crm_cols.get("grid", ""), "")
                            if match_acc is not None
                            else ""
                        )

                        # Match with Apify using Name, URL query, and Lat/Lng checks
                        if crm_label == "No CRM Match":
                            lead_name_raw = lead_row.get(lead_cols.get("name", ""), "")
                            lead_sangkat_raw = lead_row.get(lead_cols.get("sangkat", ""), "")

                            lead_lat, lead_lng = None, None
                            try:
                                raw_lat = lead_row.get(lead_cols.get("lat", ""))
                                raw_lng = lead_row.get(lead_cols.get("lng", ""))
                                if pd.notnull(raw_lat) and pd.notnull(raw_lng):
                                    lead_lat = float(raw_lat)
                                    lead_lng = float(raw_lng)
                            except (ValueError, TypeError):
                                lead_lat, lead_lng = None, None

                            apify_match_row, dist_note = match_lead_to_apify(
                                lead_name=str(lead_name_raw),
                                lead_sangkat=str(lead_sangkat_raw),
                                lead_lat=lead_lat,
                                lead_lng=lead_lng,
                                df_apify=df_apify,
                                apify_cols=apify_cols,
                                max_radius_meters=apify_proximity_radius,
                            )

                            apify_label, apify_reason = validate_apify_status(
                                apify_match_row,
                                apify_cols,
                                valid_categories_list,
                            )
                            final_label = apify_label
                            reason = f"{apify_reason} ({dist_note})"

                        res_row = lead_row.to_dict()
                        res_row["Final Classification"] = final_label
                        res_row["Classification Reason"] = reason
                        res_row["Match Score (%)"] = round(score, 1)
                        res_row["Matched CRM Account Name"] = matched_crm_name
                        res_row["Matched CRM GRID"] = matched_crm_grid
                        results.append(res_row)

                    out_df = pd.DataFrame(results)

                    st.divider()
                    st.subheader("Classification Results Summary")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Leads", len(out_df))
                    m2.metric("✅ P1 — New", len(out_df[out_df["Final Classification"] == "P1 — New"]))
                    m3.metric("🔴 P4 — Duplicate", len(out_df[out_df["Final Classification"] == "P4 — Duplicate"]))
                    m4.metric("🟡 P3 — Potential", len(out_df[out_df["Final Classification"] == "P3 — Potential Match"]))
                    m5.metric(
                        "⚪ P2 / Closed",
                        len(
                            out_df[
                                out_df["Final Classification"].isin(
                                    ["P2 — Please Check", "Business Closed", "Wrong Target Group"]
                                )
                            ]
                        ),
                    )

                    st.dataframe(out_df, use_container_width=True)

                    excel_bytes = export_to_excel(out_df)
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_bytes,
                        file_name="Cambodia_Classified_Leads_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")

with tab2:
    st.markdown("## Step 1 · Generate URLs")
    st.caption("Upload your leads file to generate search URLs for Apify.")

    step1_file = st.file_uploader(
        "Upload leads file (.xlsx or .csv)",
        type=["xlsx", "csv"],
        key="step1_upload",
    )

    if step1_file:
        df_step1 = (
            pd.read_excel(step1_file)
            if step1_file.name.endswith(".xlsx")
            else pd.read_csv(step1_file)
        )

        name_col = resolve_column(
            df_step1, ["Company / Account", "Company", "Lead Name", "Name"]
        )
        sangkat_col = resolve_column(
            df_step1,
            ["Sangkat / Khan / Province", "Sangkat", "District", "City", "Street"],
        )

        if name_col:
            generated_data = []
            for idx, row in df_step1.iterrows():
                name_val = str(row.get(name_col, "")).strip()

                sangkat_raw = row.get(sangkat_col, "") if sangkat_col else ""
                sangkat_val = (
                    ""
                    if pd.isna(sangkat_raw) or str(sangkat_raw).lower() == "nan"
                    else str(sangkat_raw).strip()
                )

                full_query = f"{name_val} {sangkat_val} Cambodia".strip()
                full_query = re.sub(r"\s+", " ", full_query)

                encoded_q = urllib.parse.quote(full_query)
                google_url = f"https://www.google.com/maps/search/?api=1&query={encoded_q}"

                generated_data.append({
                    "Company Name": name_val,
                    "Sangkat / Area": sangkat_val,
                    "Search Query": full_query,
                    "url": google_url,
                })

            df_generated = pd.DataFrame(generated_data)
            st.session_state["generated_urls_df"] = df_generated
            st.success(f"Generated {len(df_generated)} URLs.")
            st.dataframe(df_generated, use_container_width=True)

            generated_targets_text = "\n".join(df_generated["url"].tolist())
            st.text_area("Generated targets", value=generated_targets_text, height=200)

            csv_data = df_generated.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Generated URLs CSV (Upload to Apify)",
                data=csv_data,
                file_name="Apify_Generated_URLs.csv",
                mime="text/csv",
                type="primary",
            )

    st.divider()
    st.markdown("## Step 2 · Inspect Apify Results")
    st.caption("Upload your Apify scrape export here to preview categories and coordinate status before classification.")

    apify_inspect_file = st.file_uploader(
        "Upload Apify export (.csv or .xlsx)",
        type=["csv", "xlsx"],
        key="apify_inspect",
    )

    if apify_inspect_file:
        df_apify_raw = (
            pd.read_excel(apify_inspect_file)
            if apify_inspect_file.name.endswith(".xlsx")
            else pd.read_csv(apify_inspect_file)
        )
        st.session_state["loaded_apify_df"] = df_apify_raw
        st.success(f"Successfully loaded {len(df_apify_raw)} scraped Google Maps places into session memory.")
        st.dataframe(df_apify_raw.head(25), use_container_width=True)

with tab3:
    st.subheader("🏢 Salesforce CRM Internal Duplicate Audit")
    crm_audit_file = st.file_uploader(
        "Upload CRM Accounts File", type=["xlsx", "csv"], key="crm_audit"
    )

    if crm_audit_file and st.button("Run Internal Audit"):
        df_audit = (
            pd.read_excel(crm_audit_file)
            if crm_audit_file.name.endswith(".xlsx")
            else pd.read_csv(crm_audit_file)
        )
        name_col = resolve_column(df_audit, ["Account Name", "Name", "Company"])

        if name_col and HAS_RAPIDFUZZ:
            duplicates = []
            records = df_audit.head(400).to_dict("records")
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    r1, r2 = records[i], records[j]
                    n1 = normalize_cambodian_text(r1.get(name_col))
                    n2 = normalize_cambodian_text(r2.get(name_col))
                    if n1 and n2:
                        score = fuzz.token_set_ratio(n1, n2) / 100.0
                        if score >= p4_threshold:
                            duplicates.append({
                                "Account 1": r1.get(name_col),
                                "Account 2": r2.get(name_col),
                                "Similarity Score (%)": round(score * 100.0, 1),
                            })
            if duplicates:
                st.warning(f"Found {len(duplicates)} duplicate pairs!")
                st.dataframe(pd.DataFrame(duplicates), use_container_width=True)
            else:
                st.success("No duplicates found above threshold.")

with tab4:
    st.markdown("""
    ### 📖 Cambodian Lead Classifier Guide
    
    #### Classification Definitions
    - **P1 — New:** No CRM match, Apify-confirmed restaurant verified within geographic radius
    - **P2 — Please Check:** No matching Apify result found, or placed outside geographic radius
    - **P3 — Potential Match:** Name similarity 50%–74% at same location or Sangkat
    - **P4 — Duplicate:** Name similarity ≥ 75% at same location or Sangkat
    - **Business Closed:** Google confirms permanently or temporarily closed
    - **Wrong Target Group:** Google Maps category is not food-delivery eligible
    """)
