import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add fetch_reports to imports
content = content.replace(
    '    fetch_breakouts,\n)',
    '    fetch_breakouts,\n    fetch_reports,\n)'
)

# 2. Update load_all_data
content = content.replace(
    '    return (btc_summary, eth_summary,\n            btc_df, eth_df, breakouts)',
    '    daily_reports = fetch_reports("daily_reports", limit=5)\n    weekly_reports = fetch_reports("weekly_reports", limit=5)\n\n    return (btc_summary, eth_summary,\n            btc_df, eth_df, breakouts, daily_reports, weekly_reports)'
)

# 3. Update main to unpack them
content = content.replace(
    '         breakouts_df) = load_all_data()',
    '         breakouts_df, daily_reports, weekly_reports) = load_all_data()'
)

# 4. Add render_reports_section function before main
reports_section = '''
import json
def render_reports_section(daily_df, weekly_df):
    st.markdown("### 📑 Automated Reports", unsafe_allow_html=False)
    
    daily_tab, weekly_tab = st.tabs(["Daily Reports", "Weekly Reports"])
    
    with daily_tab:
        if daily_df.empty:
            st.info("No daily reports found.")
        else:
            for idx, row in daily_df.iterrows():
                with st.expander(f"Daily Report: {row['report_date']}"):
                    st.json(json.loads(row['report_data']) if isinstance(row['report_data'], str) else row['report_data'])
                    
    with weekly_tab:
        if weekly_df.empty:
            st.info("No weekly reports found.")
        else:
            for idx, row in weekly_df.iterrows():
                with st.expander(f"Weekly Report: {row['report_date']}"):
                    st.json(json.loads(row['report_data']) if isinstance(row['report_data'], str) else row['report_data'])

'''

content = content.replace(
    'def main():',
    reports_section + 'def main():'
)

# 5. Call render_reports_section in main
content = content.replace(
    '    # Footer',
    '    # Reports section\n    render_reports_section(daily_reports, weekly_reports)\n\n    st.divider()\n\n    # Footer'
)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
