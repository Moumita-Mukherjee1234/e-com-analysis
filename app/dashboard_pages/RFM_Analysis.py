import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pycountry
from utils.style import style_plotly


def show_rfm(customer_features, retail_filtered):

    # ====================================
    # PAGE HEADER
    # ====================================

    st.markdown("""
    <h1 style="
        color:#111827;
        font-size:40px;
        font-weight:900;
        margin-bottom:0;
    ">
        📦 RFM ANALYSIS
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        font-size:18px;
        color:#64748B;
        margin-bottom:25px;
    ">
        Customer segmentation using Recency, Frequency and Monetary metrics
    </div>
    """, unsafe_allow_html=True)

    # ====================================
    # LOAD CUSTOMER FEATURES
    # ====================================

    if "uploaded_customer_features" in st.session_state:

        rfm_data = (
            st.session_state[
                "uploaded_customer_features"
            ]
            .copy()
        )

    else:

        rfm_data = customer_features.copy()

    # ====================================
    # KPI ROW
    # ====================================

    avg_recency = round(
        rfm_data["recency_days"].mean(),
        1
    )

    avg_frequency = round(
        rfm_data["frequency"].mean(),
        1
    )

    avg_monetary = round(
        rfm_data["monetary"].mean(),
        2
    )

    total_customers = len(rfm_data)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "👥 Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "📅 Avg Recency",
            avg_recency
        )

    with c3:
        st.metric(
            "🔄 Avg Frequency",
            avg_frequency
        )

    with c4:
        st.metric(
            "💰 Avg Monetary",
            f"₹{avg_monetary:,.0f}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================
    # GEOGRAPHY ANALYTICS MAP
    # ====================================

    

    retail_data = retail_filtered.copy()

    if (
        "Country" in retail_data.columns
        and "total_price" in retail_data.columns
    ):

        # Top 10 countries by revenue
        country_revenue = (
            retail_data
            .groupby("Country")["total_price"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
    
        # Convert country names to ISO-3 codes
        def get_country_code(country_name):
    
            try:
                return pycountry.countries.lookup(
                    country_name
                ).alpha_3
    
            except Exception:
                return None
    
        country_revenue["iso_alpha"] = (
            country_revenue["Country"]
            .apply(get_country_code)
        )
    
        country_revenue = (
            country_revenue
            .dropna(subset=["iso_alpha"])
        )
    
        fig_geo = px.choropleth(
            country_revenue,
            locations="iso_alpha",
            color="total_price",
            hover_name="Country",
            color_continuous_scale="Reds",
            title="🌍 Top 10 Countries by Revenue"
        )
    
        fig_geo.update_geos(
            showframe=False,
            showcoastlines=True,
            showcountries=True,
            countrycolor="white",
            showland=True,
            landcolor="#FFFFFF",
            projection_type="natural earth"
        )
    
        fig_geo.update_layout(
            height=500,
    
            width=950,  # reduce map width
    
            margin=dict(
                l=80,
                r=80,
                t=60,
                b=20
            ),
    
            coloraxis_colorbar=dict(
                title="Revenue",
                x=0.02,      # move legend left
                y=0.5,
                len=0.70,
                thickness=18
            )
        )

        fig_geo = style_plotly(fig_geo)

        # Center align map
        c1, c2, c3 = st.columns([1.93, 5, 1])
    
        with c2:
            st.plotly_chart(
                fig_geo,
                use_container_width=False
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================
    # RFM VISUALIZATIONS
    # ====================================

    left, right = st.columns(2)

    with left:

        hist_col = (
            "rfm_score"
            if "rfm_score" in rfm_data.columns
            else "monetary"
        )

        fig1 = px.histogram(
            rfm_data,
            x=hist_col,
            nbins=30,
            title=f"{hist_col.upper()} Distribution",
            color_discrete_sequence=["#8A6ED7"]
        )

        fig1.update_layout(
            height=450
        )

        fig1 = style_plotly(fig1)

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with right:

        rfm_plot = rfm_data.copy()

        rfm_plot["monetary_size"] = np.log1p(
            rfm_plot["monetary"].abs()
        )

        fig2 = px.scatter(
            rfm_plot,
            x="recency_days",
            y="frequency",
            size="monetary_size",
            size_max=45,
            hover_data=["Customer ID"],
            title="Recency vs Frequency (Bubble Size = Spend)",
            color="monetary",
            color_continuous_scale="Purples"
        )

        fig2.update_layout(
            height=450
        )

        fig2 = style_plotly(fig2)

        st.plotly_chart(
            fig2,
            use_container_width=True
        )