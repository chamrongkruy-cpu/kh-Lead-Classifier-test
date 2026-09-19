with tab2:
    st.markdown("## Step 1 · Generate URLs")
    st.caption("Upload your leads file to generate search URLs for Apify.")

    step1_file = st.file_uploader("Upload leads file (.xlsx or .csv)", type=["xlsx", "csv"], key="step1_upload")

    if step1_file:
        df_step1 = pd.read_excel(step1_file) if step1_file.name.endswith('.xlsx') else pd.read_csv(step1_file)
        
        grid_col = resolve_column(df_step1, ['GRID', 'Lead ID', 'Id'])
        name_col = resolve_column(df_step1, ['Company / Account', 'Company', 'Lead Name', 'Name'])
        sangkat_col = resolve_column(df_step1, ['Sangkat / Khan / Province', 'Sangkat', 'District', 'City', 'Street'])

        if name_col:
            generated_data = []
            for idx, row in df_step1.iterrows():
                grid_val = row.get(grid_col, f"GRID_{idx}") if grid_col else f"GRID_{idx}"
                name_val = str(row.get(name_col, '')).strip()
                sangkat_val = str(row.get(sangkat_col, '')).strip() if sangkat_col else ""
                
                search_term = f"{name_val} {sangkat_val} Cambodia".strip()
                encoded_q = re.sub(r'\s+', '+', search_term)
                google_url = f"https://www.google.com/maps/search/{encoded_q}"
                
                generated_data.append({
                    "GRID": grid_val,
                    "Company Name": name_val,
                    "Search Query": search_term,
                    "url": google_url
                })
            
            df_generated = pd.DataFrame(generated_data)
            st.session_state['generated_urls_df'] = df_generated
            st.success(f"Generated {len(df_generated)} URLs.")
            st.dataframe(df_generated, use_container_width=True)

            csv_data = df_generated.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Generated URLs CSV (Upload to Apify)",
                data=csv_data,
                file_name="Apify_Generated_URLs.csv",
                mime="text/csv",
                type="primary"
            )

    st.divider()
    st.markdown("## Step 2 · Add GRID to your Apify Export")
    st.caption("After running Apify, upload your export here. The tool matches each row via `inputUrl` and adds a `GRID` column.")

    if 'generated_urls_df' not in st.session_state:
        st.info("No URLs generated this session yet. Upload your URL CSV below if generated in a previous session.")
        prev_url_file = st.file_uploader("Upload URL CSV (from previous session)", type=["csv", "xlsx"], key="prev_urls")
        if prev_url_file:
            st.session_state['generated_urls_df'] = pd.read_excel(prev_url_file) if prev_url_file.name.endswith('.xlsx') else pd.read_csv(prev_url_file)

    apify_export_file = st.file_uploader("Upload Apify export (.csv or .xlsx)", type=["csv", "xlsx"], key="apify_export")

    if apify_export_file:
        if 'generated_urls_df' in st.session_state:
            df_gen_urls = st.session_state['generated_urls_df']
            df_apify_raw = pd.read_excel(apify_export_file) if apify_export_file.name.endswith('.xlsx') else pd.read_csv(apify_export_file)

            # Extended column lookup to handle various Apify export formats
            input_url_col = resolve_column(df_apify_raw, ['inputUrl', 'searchUrl', 'url', 'input_url', 'startUrl', 'query', 'url/url', 'input/url', 'Search Query'])
            
            if input_url_col:
                gen_url_col = resolve_column(df_gen_urls, ['url', 'Google Maps Search URL', 'Search Query'])
                gen_grid_col = resolve_column(df_gen_urls, ['GRID'])

                if gen_url_col and gen_grid_col:
                    # Merge Apify export with generated URLs matching on URL or Query
                    merged_df = pd.merge(
                        df_apify_raw,
                        df_gen_urls[[gen_url_col, gen_grid_col]],
                        left_on=input_url_col,
                        right_on=gen_url_col,
                        how='left'
                    )
                    st.success("Successfully matched and added GRID column!")
                    st.dataframe(merged_df, use_container_width=True)

                    enriched_csv = merged_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Enriched Apify Export with GRID",
                        data=enriched_csv,
                        file_name="Apify_Export_With_GRID.csv",
                        mime="text/csv",
                        type="primary"
                    )
                else:
                    st.error("The reference URLs dataframe is missing the 'url' or 'GRID' column.")
            else:
                st.error("Could not find `inputUrl` in the uploaded Apify export file. Available columns in your uploaded file are: " + ", ".join(list(df_apify_raw.columns[:10])))
        else:
            st.warning("Please upload or generate URLs first before attaching GRID to Apify export.")
