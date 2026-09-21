import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Tourism Experience Analytics Platform")
st.markdown(
    "Predict travel behavior, receive personalized attraction recommendations, and explore tourism trends."
)


# ---------------------------------------------------------
# 1. Load Saved DataFrames & Models (Cached for Performance)
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    # Load Saved DataFrames
    consolidated_df = pd.read_pickle("consolidated_df.pkl")
    df_item = pd.read_pickle("item.pkl")
    df_city = pd.read_pickle("city.pkl")

    # Load Models and Encoders
    classifier_model = joblib.load("classifier_model.pkl")
    visit_mode_encoder = joblib.load("visit_mode_encoder.pkl")
    recommender = joblib.load("collaborative_recommender.pkl")

    return (
        consolidated_df,
        df_item,
        df_city,
        classifier_model,
        visit_mode_encoder,
        recommender,
    )


try:
    (
        consolidated_df,
        df_item,
        df_city,
        classifier_model,
        visit_mode_encoder,
        recommender,
    ) = load_assets()
except Exception as e:
    st.error(
        f"Error loading saved assets. Please ensure all `.pkl` files are present in the directory.\n\nDetails: {e}"
    )
    st.stop()


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "🔮 Visit Mode Prediction",
        "🎯 Personalized Recommendations",
        "📊 Tourism Analytics Dashboard",
    ]
)


# =========================================================
# TAB 1: Visit Mode Prediction
# =========================================================
with tab1:
    st.header("Predict User Visit Mode")
    st.write(
        "Input user demographic details and travel specifics to predict the likely visit mode (e.g., Family, Business, Couples)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Location")
        continent_options = (
            consolidated_df["Continent"].dropna().unique().tolist()
        )
        selected_continent = st.selectbox("Select Continent", continent_options)

        # Filter countries based on continent
        filtered_countries = (
            consolidated_df[
                consolidated_df["Continent"] == selected_continent
            ]["Country"]
            .dropna()
            .unique()
            .tolist()
        )
        selected_country = st.selectbox("Select Country", filtered_countries)

        # Filter cities based on country
        filtered_cities = (
            consolidated_df[
                consolidated_df["Country"] == selected_country
            ]["CityName"]
            .dropna()
            .unique()
            .tolist()
        )
        selected_city = st.selectbox("Select Residence City", filtered_cities)

    with col2:
        st.subheader("Visit Attributes")
        visit_year = st.slider("Visit Year", 2015, 2026, 2023)
        visit_month = st.slider("Visit Month", 1, 12, 6)

        attraction_type_options = (
            df_item["AttractionType"].dropna().unique().tolist()
        )
        selected_attr_type = st.selectbox(
            "Target Attraction Type", attraction_type_options
        )

    # Predict Button Execution
    if st.button("Predict Visit Mode", type="primary"):
        # Map selected categorical options back to IDs from saved consolidated_df
        try:
            cont_id = consolidated_df[
                consolidated_df["Continent"] == selected_continent
            ]["ContinentId"].iloc[0]
            country_id = consolidated_df[
                consolidated_df["Country"] == selected_country
            ]["CountryId"].iloc[0]
            region_id = consolidated_df[
                consolidated_df["Country"] == selected_country
            ]["RegionId"].iloc[0]
            city_id = consolidated_df[
                consolidated_df["CityName"] == selected_city
            ]["CityId"].iloc[0]
            attr_type_id = df_item[
                df_item["AttractionType"] == selected_attr_type
            ]["AttractionTypeId"].iloc[0]

            # Construct feature vector expected by classifier
            input_features = pd.DataFrame(
                [
                    {
                        "ContinentId": cont_id,
                        "RegionId": region_id,
                        "CountryId": country_id,
                        "CityId": city_id,
                        "VisitYear": visit_year,
                        "VisitMonth": visit_month,
                        "AttractionTypeId": attr_type_id,
                        "AttractionCityId": city_id,
                    }
                ]
            )

            # Predict numeric class and encode back to text label
            pred_numeric = classifier_model.predict(input_features)[0]
            predicted_mode = visit_mode_encoder.inverse_transform(
                [pred_numeric]
            )[0]

            st.success(f"🎉 **Predicted Visit Mode:** `{predicted_mode}`")

        except Exception as err:
            st.error(
                f"Error processing inputs for prediction. Ensure feature IDs map correctly: {err}"
            )


# =========================================================
# TAB 2: Personalized Recommendations
# =========================================================
with tab2:
    st.header("Attraction Recommendations")
    st.write(
        "Select an existing user profile or enter preferences to view top recommended tourist attractions."
    )

    # Select existing User ID from consolidated dataset
    user_list = sorted(consolidated_df["UserId"].unique())
    selected_user = st.selectbox("Select User ID Profile", user_list)

    top_n_recs = st.slider(
        "Number of Recommendations", 3, 10, 5, key="rec_slider"
    )

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Calculating similarity matrix..."):
            try:
                recommendations_df = recommender.recommend(
                    user_id=selected_user,
                    top_n=top_n_recs,
                    item_metadata_df=df_item,
                )

                st.subheader(
                    f"Top {top_n_recs} Recommended Attractions for User `{selected_user}`:"
                )

                # Format results for display
                display_cols = [
                    c
                    for c in [
                        "AttractionId",
                        "Attraction",
                        "AttractionType",
                        "CityName",
                        "PredictedScore",
                    ]
                    if c in recommendations_df.columns
                ]
                st.dataframe(
                    recommendations_df[display_cols].style.highlight_max(
                        axis=0, color="#d1e7dd"
                    ),
                    use_container_width=True,
                )

            except Exception as err:
                st.warning(
                    f"Could not calculate collaborative recommendations: {err}"
                )


# =========================================================
# TAB 3: Tourism Analytics Dashboard
# =========================================================
with tab3:
    st.header("📊 Tourism Analytics & Popularity Dashboard")
    st.write(
        "Explore trends in popular attractions, top geographic regions, and demographic user segments."
    )

    # ---------------------------------------------------------
    # 1. Popular Attractions Analysis
    # ---------------------------------------------------------
    st.subheader("🏆 Top Popular Attractions")

    col_att1, col_att2 = st.columns([1, 2])

    with col_att1:
        min_reviews = st.slider(
            "Filter Minimum Review Count Threshold", 1, 50, 5
        )
        top_k_attractions = st.slider(
            "Select Number of Top Attractions", 5, 20, 10
        )

    # Aggregate Attraction Metrics
    attraction_col = (
        "Attraction" if "Attraction" in consolidated_df.columns else "AttractionId"
    )

    attraction_stats = (
        consolidated_df.groupby(attraction_col)
        .agg(
            Total_Visits=("Rating", "count"),
            Average_Rating=("Rating", "mean"),
        )
        .reset_index()
    )

    # Filter by minimum reviews threshold
    filtered_attractions = (
        attraction_stats[attraction_stats["Total_Visits"] >= min_reviews]
        .sort_values(by="Total_Visits", ascending=False)
        .head(top_k_attractions)
    )

    with col_att2:
        fig_pop_attractions = px.bar(
            filtered_attractions,
            x="Total_Visits",
            y=attraction_col,
            orientation="h",
            color="Average_Rating",
            title=f"Top {top_k_attractions} Most Visited Attractions (Min {min_reviews} Reviews)",
            labels={
                "Total_Visits": "Total Visits / Reviews",
                attraction_col: "Attraction",
                "Average_Rating": "Avg Rating",
            },
            color_continuous_scale="Viridis",
        )
        fig_pop_attractions.update_layout(
            yaxis={"categoryorder": "total ascending"}, height=400
        )
        st.plotly_chart(fig_pop_attractions, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 2. Top Geographic Regions
    # ---------------------------------------------------------
    st.subheader("🌍 Regional Hotspots")

    col_reg1, col_reg2 = st.columns(2)

    with col_reg1:
        top_regions_df = (
            consolidated_df.groupby("Region")
            .agg(
                User_Visits=("Rating", "count"),
                Mean_Rating=("Rating", "mean"),
            )
            .reset_index()
            .sort_values(by="User_Visits", ascending=False)
            .head(10)
        )

        fig_regions = px.bar(
            top_regions_df,
            x="Region",
            y="User_Visits",
            color="Mean_Rating",
            title="Top 10 Tourist Regions by Visit Volume",
            labels={
                "User_Visits": "Number of Visits",
                "Mean_Rating": "Avg Rating",
            },
            color_continuous_scale="Plasma",
        )
        fig_regions.update_layout(height=400)
        st.plotly_chart(fig_regions, use_container_width=True)

    with col_reg2:
        # Rating Distribution Boxplot Across Top Regions
        top_10_region_names = top_regions_df["Region"].tolist()
        df_top_regions = consolidated_df[
            consolidated_df["Region"].isin(top_10_region_names)
        ]

        fig_region_box = px.box(
            df_top_regions,
            x="Region",
            y="Rating",
            color="Region",
            title="Rating Distribution Across Top 10 Regions",
        )
        fig_region_box.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_region_box, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 3. User Segmentation & Demographic Hierarchy
    # ---------------------------------------------------------
    st.subheader("👥 User Segment Demographics")

    st.write(
        "Interactive breakdown of user segments across Continents, Regions, and Countries."
    )

    # Clean missing values for Sunburst
    df_sunburst = consolidated_df.drop_duplicates(subset=["UserId"])[
        ["Continent", "Region", "Country"]
    ].copy()
    df_sunburst["Continent"] = df_sunburst["Continent"].fillna(
        "Unknown Continent"
    )
    df_sunburst["Region"] = df_sunburst["Region"].fillna("Unknown Region")
    df_sunburst["Country"] = df_sunburst["Country"].fillna("Unknown Country")

    fig_sunburst = px.sunburst(
        df_sunburst,
        path=["Continent", "Region", "Country"],
        title="User Segment Hierarchy (Click to Expand: Continent -> Region -> Country)",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig_sunburst.update_layout(height=550)
    st.plotly_chart(fig_sunburst, use_container_width=True)