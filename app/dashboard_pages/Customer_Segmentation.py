import streamlit as st
import pandas as pd
import plotly.express as px

from utils.style import style_plotly


def show_customer_segmentation(
    customer_features,
    customer_clusters
):

    # ==================================
    # CUSTOM TABLE FUNCTION
    # ==================================

    def show_centered_table(df):

        html = f"""
        <div style="overflow-x:auto;">
            {df.to_html(
                index=False,
                classes="custom-table"
            )}
        </div>
        """

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    # ==================================
    # PAGE HEADER
    # ==================================

    st.markdown("""
    <h1 style="
        color:#111827;
        font-size:42px;
        font-weight:900;
        margin-bottom:0;
    ">
        🧩 CUSTOMER ANALYSIS
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        font-size:18px;
        color:#64748B;
        margin-top:-10px;
        margin-bottom:30px;
    ">
        Analyze customer groups based on purchasing behavior,
        frequency, recency and spending patterns
    </div>
    """, unsafe_allow_html=True)

    # ==================================
    # DATA SOURCE
    # ==================================

    if "uploaded_predictions" in st.session_state:

        merged = (
            st.session_state[
                "uploaded_predictions"
            ]
            .copy()
        )

       

    else:

        merged = customer_features.merge(
            customer_clusters,
            on="Customer ID",
            how="left"
        )

    # ==================================
    # BUSINESS SEGMENTS
    # ==================================

    freq_80 = merged["frequency"].quantile(0.80)
    freq_70 = merged["frequency"].quantile(0.70)
    freq_60 = merged["frequency"].quantile(0.60)

    mon_80 = merged["monetary"].quantile(0.80)

    rec_20 = merged["recency_days"].quantile(0.20)
    rec_80 = merged["recency_days"].quantile(0.80)
    rec_90 = merged["recency_days"].quantile(0.90)


    def assign_segment(row):

        if (
            row["frequency"] >= freq_80
            and row["monetary"] >= mon_80
            and row["recency_days"] <= rec_20
        ):
            return "VIP"
    
        elif row["frequency"] >= freq_70:
            return "Loyal"
    
        elif (
            row["frequency"] >= freq_60
            and row["recency_days"] >= rec_80
        ):
            return "At Risk"
    
        elif row["recency_days"] >= rec_90:
            return "Lost"
    
        else:
            return "Regular"


    merged["segment"] = merged.apply(
        assign_segment,
        axis=1
    )

    segment_col = "segment"      

    # ==================================
    # KPI VALUES
    # ==================================

    total_customers = len(merged)

    total_segments = (
        merged[segment_col]
        .nunique()
    )

    avg_frequency = round(
        merged["frequency"].mean(),
        2
    )

    avg_monetary = round(
        merged["monetary"].mean(),
        2
    )

    # ==================================
    # KPI CARDS
    # ==================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">
                👥 Total Customers
            </div>
            <div class="kpi-value">
                {total_customers:,}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:

        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">
                🎯 Segments
            </div>
            <div class="kpi-value">
                {total_segments}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:

        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">
                🔄 Avg Frequency
            </div>
            <div class="kpi-value">
                {avg_frequency}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:

        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">
                💰 Avg Monetary
            </div>
            <div class="kpi-value">
                ₹{avg_monetary:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div style='height:30px'></div>",
        unsafe_allow_html=True
    )

    # ==================================
    # SEGMENT COUNTS
    # ==================================

    cluster_counts = (
        merged[segment_col]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Segment",
        "Customers"
    ]

    segment_colors = [
        "#7C3AED",
        "#3B82F6",
        "#22C55E",
        "#F59E0B",
        "#EF4444"
    ]

    # ==================================
    # DONUT CHART
    # ==================================

    fig_donut = px.pie(
        cluster_counts,
        names="Segment",
        values="Customers",
        hole=0.60,
        title="Customer Segment Distribution",
        color_discrete_sequence=segment_colors
    )


    fig_donut.update_layout(
        showlegend=True,
        legend=dict(
        orientation="h",
        y=-0.15,
        x=0.5,
        xanchor="center"
        ),
        margin=dict(
            t=60,
            b=80,
            l=20,
            r=20
        )
    )


    fig_donut = style_plotly(
        fig_donut
    )

    # ==================================
    # SCATTER PLOT
    # ==================================

    fig_scatter = px.scatter(
        merged,
        x="frequency",
        y="monetary",
        color=segment_col,
        hover_data=[
            "Customer ID",
            "recency_days"
        ],
        title="Frequency vs Monetary",
        opacity=0.85,
        color_discrete_sequence=segment_colors
    )

    fig_scatter.update_traces(
        marker=dict(
            size=10,
            line=dict(
                width=1,
                color="white"
            )
        )
    )

    fig_scatter.update_layout(
        title_x=0,     
        title_y=0.95,  
        xaxis_title="Frequency",
        yaxis_title="Monetary Value",

        legend=dict(
            orientation="v",
            x=0.8,
            y=1,
            xanchor="left",
            yanchor="bottom"
        ),
         margin=dict(
            t=40,
            b=40,
            l=20,
            r=20
        )
    )

    fig_scatter = style_plotly(
        fig_scatter
    )

    # ==================================
    # ROW 1
    # DONUT + SCATTER
    # ==================================

    left, right = st.columns([1, 2])

    with left:

        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )

    with right:

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    st.markdown(
        "<div style='height:30px'></div>",
        unsafe_allow_html=True
    )
        # ==================================
    # CUSTOMERS PER SEGMENT
    # ==================================

    fig_bar = px.bar(
        cluster_counts,
        x="Segment",
        y="Customers",
        color="Segment",
        text="Customers",
        title="Customers per Segment",
        color_discrete_sequence=segment_colors
    )


    fig_bar.update_layout(
        xaxis_title="Segment",
        yaxis_title="Number of Customers",
        showlegend=False
    )

    fig_bar = style_plotly(
        fig_bar
    )

    # ==================================
    # REVENUE CONTRIBUTION
    # ==================================

    revenue_segment = (
        merged
        .groupby(segment_col)["monetary"]
        .sum()
        .reset_index()
    )

    revenue_segment.columns = [
        "Segment",
        "Revenue"
    ]

    fig_revenue = px.pie(
        revenue_segment,
        names="Segment",
        values="Revenue",
        title="Revenue Contribution by Segment",
        color_discrete_sequence=segment_colors
    )
    fig_revenue = style_plotly(
    fig_revenue
    )

    fig_revenue.update_layout(
        title_x=0,     
        title_y=0.95,  
        xaxis_title="Frequency",
        yaxis_title="Monetary Value",

    legend=dict(
        orientation="v",
        x=0.8,
        y=1,
        xanchor="left",
        yanchor="bottom"
    ),
    margin=dict(
        t=40,
        b=40,
        l=20,
        r=20
    )
)




    # ==================================
    # ROW 2
    # BAR + REVENUE PIE
    # ==================================

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    with right:

        st.plotly_chart(
            fig_revenue,
            use_container_width=True
        )

    st.markdown(
        "<div style='height:30px'></div>",
        unsafe_allow_html=True
    )

    # ==================================
    # TOP CUSTOMERS
    # ==================================

    st.subheader(
        "🏆 Top 10 Highest Value Customers"
    )

    top_customers = (
        merged[
            [
                "Customer ID",
                segment_col,
                "frequency",
                "monetary",
                "recency_days"
            ]
        ]
        .sort_values(
            "monetary",
            ascending=False
        )
        .head(10)
    )

    top_customers.columns = [
        "Customer ID",
        "Segment",
        "Frequency",
        "Monetary",
        "Recency"
    ]

    show_centered_table(
        top_customers
    )

    st.markdown(
        "<div style='height:30px'></div>",
        unsafe_allow_html=True
    )

    # ==================================
    # INSIGHTS
    # ==================================

    segment_summary = (
        merged
        .groupby(segment_col)
        .agg(
            Customers=(
                "Customer ID",
                "count"
            ),
            Avg_Recency=(
                "recency_days",
                "mean"
            ),
            Avg_Frequency=(
                "frequency",
                "mean"
            ),
            Avg_Monetary=(
                "monetary",
                "mean"
            )
        )
        .round(2)
        .reset_index()
    )

    highest_value_segment = (
        segment_summary
        .sort_values(
            "Avg_Monetary",
            ascending=False
        )
        .iloc[0][segment_col]
    )

    most_active_segment = (
        segment_summary
        .sort_values(
            "Avg_Frequency",
            ascending=False
        )
        .iloc[0][segment_col]
    )

    most_recent_segment = (
        segment_summary
        .sort_values(
            "Avg_Recency",
            ascending=True
        )
        .iloc[0][segment_col]
    )

    highest_revenue_segment = (
        revenue_segment
        .sort_values(
            "Revenue",
            ascending=False
        )
        .iloc[0]["Segment"]
    )

    st.subheader(
        "💡 Key Insights"
    )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#C4B5FD,#A78BFA);
            padding:20px;
            border-radius:18px;
            color:white;
            font-size:16px;
            line-height:0.98;
            box-shadow:0 10px 25px rgba(34,197,94,0.25);
        ">

        🏆 <b>Highest Revenue Segment:</b> {highest_revenue_segment}<br><br>

        💰 <b>Highest Value Customers</b> belong primarily to the
        <b>{highest_value_segment}</b> segment.<br><br>
    
       🔄 <b>Most Active Segment:</b> {most_active_segment}<br><br>
    
        🛍️ Customers in this segment purchase most frequently.<br><br>
    
        ⚡ <b>Most Recent Buyers:</b> {most_recent_segment}<br><br>
    
        📈 Revenue is concentrated in high-value customer groups.<br><br>
    
        🎯 Focus retention campaigns, loyalty programs and personalized recommendations on VIP and Loyal customers.
    
        </div>
        """,
        unsafe_allow_html=True
    )