from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:  # pragma: no cover - altair optional for richer charts
    alt = None

from heuristic_ranking import RankingConfig, rank_candidates


def parse_connection(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().replace(",", "")
    if text.endswith("+"):
        text = text[:-1]
    return int(text) if text.isdigit() else 0


@st.cache_data(show_spinner=False)
def load_demo_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"id", "job_title", "location", "connection"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["connection_numeric"] = df["connection"].apply(parse_connection)
    return df


def main() -> None:
    st.set_page_config(page_title="Potential Talent Finder", layout="wide")

    st.markdown(
        """
        <style>
        .main .block-container{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .metric-card{
            background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            border: 1px solid #e5e8eb;
            box-shadow: 0px 2px 6px rgba(15, 23, 42, 0.08);
        }
        .metric-card h3{
            font-size: 0.9rem;
            color: #475569;
            margin-bottom: 0.35rem;
        }
        .metric-card p{
            font-size: 1.3rem;
            font-weight: 600;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Potential Talent Finder")
    st.write("Adjust the search inputs and constraints to generate a short list of matching candidates.")

    st.sidebar.header("Data source")
    data_mode = st.sidebar.radio("Choose dataset", ("Demo dataset", "Upload CSV"))

    candidate_df: Optional[pd.DataFrame] = None
    data_label = ""

    if data_mode == "Demo dataset":
        data_path = Path("potential-talents - Aspiring human resources - seeking human resources.csv")
        if not data_path.exists():
            st.error(f"Could not find the demo candidate CSV at `{data_path}`.")
            return
        demo_df = load_demo_dataset(data_path)
        try:
            candidate_df = prepare_dataset(demo_df)
            data_label = "Demo dataset"
        except ValueError as exc:
            st.error(f"Demo dataset is missing required columns: {exc}")
            return
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV with columns: id, job_title, location, connection.")
            return
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return
        try:
            candidate_df = prepare_dataset(user_df)
            data_label = uploaded_file.name
        except ValueError as exc:
            st.error(f"The uploaded dataset is missing required columns: {exc}")
            return

    st.sidebar.header("Search")

    default_keywords = ["aspiring human resources", "seeking human resources"]
    keyword_text = st.sidebar.text_area(
        "Keywords (one per line)",
        value="\n".join(default_keywords),
        height=100,
    )
    keywords: List[str] = [line.strip() for line in keyword_text.splitlines() if line.strip()]

    location_options = sorted(candidate_df["location"].dropna().unique())
    selected_locations = st.sidebar.multiselect(
        "Preferred locations (optional)",
        options=location_options,
    )

    max_connections = int(candidate_df["connection_numeric"].max()) if not candidate_df.empty else 0
    min_connections = st.sidebar.slider(
        "Minimum connections",
        min_value=0,
        max_value=max(100, max_connections),
        value=0,
        step=10,
    )

    result_count = st.sidebar.slider("Number of candidates to show", 5, 50, 20, 5)

    st.sidebar.markdown("---")
    with st.sidebar.expander("Advanced & feedback", expanded=False):
        adjust_weights = st.checkbox("Adjust ranking weights", value=False)
        if adjust_weights:
            weight_similarity = st.slider(
                "Weight: title similarity", 0.0, 1.0, 0.7, 0.05, key="weight_similarity"
            )
            weight_connections = st.slider(
                "Weight: connections", 0.0, 1.0, 0.2, 0.05, key="weight_connections"
            )
            weight_location = st.slider(
                "Weight: location", 0.0, 1.0, 0.1, 0.05, key="weight_location"
            )
        else:
            weight_similarity = 0.7
            weight_connections = 0.2
            weight_location = 0.1

        starred_ids_input = st.text_input(
            "Starred candidate IDs (comma separated)",
            value="",
            placeholder="e.g. 17, 46",
        )

    if not keywords:
        st.warning("Enter at least one keyword to rank candidates.")
        return

    config = RankingConfig(
        keywords=keywords,
        weight_similarity=weight_similarity,
        weight_connections=weight_connections,
        weight_location=weight_location,
        preferred_locations=selected_locations or None,
    )

    ranked_df = rank_candidates(candidate_df, config)

    filtered_df = ranked_df.copy()
    if selected_locations:
        filtered_df = filtered_df[filtered_df["location"].isin(selected_locations)]
    if min_connections > 0:
        filtered_df = filtered_df[filtered_df["connection_numeric"] >= min_connections]

    display_df = filtered_df[["id", "job_title", "location", "connection", "fit"]]

    st.subheader(f"Top candidates ({data_label})")
    if display_df.empty:
        st.info("No candidates match the current constraints. Adjust your filters and try again.")
    else:
        shortlisted = display_df.head(result_count)

        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            top_candidate = shortlisted.iloc[0] if not shortlisted.empty else None
            avg_fit = shortlisted["fit"].mean() if not shortlisted.empty else 0.0
            avg_connections = filtered_df["connection_numeric"].mean() if not filtered_df.empty else 0
            unique_locations = filtered_df["location"].nunique() if not filtered_df.empty else 0

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>Top Candidate</h3>
                        <p>{top_candidate['job_title'] if top_candidate is not None else '—'}</p>
                        <span style="color:#6366f1;font-weight:500;">Fit score: {top_candidate['fit']:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>Average Fit</h3>
                        <p>{avg_fit:.2f}</p>
                        <span style="color:#6366f1;font-weight:500;">Across shortlist</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>Coverage</h3>
                        <p>{unique_locations} locations</p>
                        <span style="color:#6366f1;font-weight:500;">Avg connections: {avg_connections:.0f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        tab_shortlist, tab_insights = st.tabs(["Shortlist", "Insights"])

        with tab_shortlist:
            st.dataframe(shortlisted, use_container_width=True)
            csv_buffer = StringIO()
            shortlisted.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download shortlist as CSV",
                data=csv_buffer.getvalue(),
                file_name="shortlisted_candidates.csv",
                mime="text/csv",
            )

        with tab_insights:
            chart_cols = st.columns(2)
            with chart_cols[0]:
                loc_counts = (
                    filtered_df["location"]
                    .value_counts()
                    .head(10)
                    .sort_values(ascending=True)
                )
                if not loc_counts.empty:
                    loc_df = loc_counts.reset_index()
                    loc_df.columns = ["location", "count"]
                    if alt:
                        bar_chart = (
                            alt.Chart(loc_df)
                            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                            .encode(
                                x=alt.X("count:Q", title="Candidates"),
                                y=alt.Y("location:N", sort=alt.SortField("count", order="ascending"), title="Location"),
                                tooltip=["location:N", "count:Q"],
                                color=alt.value("#6366f1"),
                            )
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.bar_chart(loc_counts)
                else:
                    st.write("No location data available for charting.")
            with chart_cols[1]:
                if not filtered_df.empty:
                    scatter_df = filtered_df[["connection_numeric", "fit", "job_title"]].copy()
                    scatter_df.rename(columns={"connection_numeric": "connections"}, inplace=True)
                    if alt:
                        scatter_chart = (
                            alt.Chart(scatter_df)
                            .mark_circle(size=80, opacity=0.7)
                            .encode(
                                x=alt.X("connections:Q", title="Connections"),
                                y=alt.Y("fit:Q", title="Fit score"),
                                tooltip=["job_title:N", "connections:Q", "fit:Q"],
                                color=alt.value("#22c55e"),
                            )
                            .interactive()
                        )
                        st.altair_chart(scatter_chart, use_container_width=True)
                    else:
                        st.scatter_chart(scatter_df, x="connections", y="fit")
                else:
                    st.write("No fit scores available for charting.")

    starred_ids = [
        int(token.strip())
        for token in starred_ids_input.split(",")
        if token.strip().isdigit()
    ]

    if starred_ids:
        starred_config = RankingConfig(
            keywords=keywords,
            weight_similarity=weight_similarity,
            weight_connections=weight_connections,
            weight_location=weight_location,
            preferred_locations=selected_locations or None,
            starred_ids=starred_ids,
        )
        reranked_df = rank_candidates(candidate_df, starred_config)

        filtered_reranked = reranked_df.copy()
        if selected_locations:
            filtered_reranked = filtered_reranked[filtered_reranked["location"].isin(selected_locations)]
        if min_connections > 0:
            filtered_reranked = filtered_reranked[filtered_reranked["connection_numeric"] >= min_connections]

        display_reranked = filtered_reranked[["id", "job_title", "location", "connection", "fit"]]

        st.subheader("Updated ranking with your starred feedback")
        if display_reranked.empty:
            st.info("No candidates remain after applying the feedback with current constraints.")
        else:
            st.caption(f"Starred candidate IDs: {starred_ids}")
            reranked_shortlist = display_reranked.head(result_count)
            st.dataframe(reranked_shortlist, use_container_width=True)
            csv_buffer = StringIO()
            reranked_shortlist.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download re-ranked shortlist",
                data=csv_buffer.getvalue(),
                file_name="shortlisted_candidates_reranked.csv",
                mime="text/csv",
            )
    else:
        st.caption("Add starred candidate IDs to see how the ranking adapts to your feedback.")

    st.caption("Tip: use Streamlit’s download button in the table menu to export the shortlist.")


if __name__ == "__main__":
    main()
