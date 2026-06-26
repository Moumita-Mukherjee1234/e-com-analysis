import streamlit as st
import pandas as pd
import plotly.express as px
from utils.style import style_plotly


def show_overview(retail_filtered, customer_features):

    def section_space(px=35):
        st.markdown(
            f"<div style='margin-top:{px}px;'></div>",
            unsafe_allow_html=True
        )
    # ====================================
    # USE UPLOADED DATA IF AVAILABLE
    # ====================================
    if "uploaded_retail" in st.session_state:

        retail_filtered = st.session_state[
            "uploaded_retail"
        ].copy()

        # Create InvoiceMonth if missing
        if "InvoiceMonth" not in retail_filtered.columns:

            retail_filtered["InvoiceMonth"] = (
                pd.to_datetime(
                    retail_filtered["InvoiceDate"]
                )
                .dt.to_period("M")
                .astype(str)
            )

    # ====================================
    # PAGE HEADER
    # ====================================

    st.markdown("""
    <h1 style="
    color:#111827;
    font-size:42px;
    font-weight:900;
    margin-bottom:0;
    ">
    📊 OVERVIEW
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="
            font-size:18px;
            color:#64748B;
            margin-top:-10px;
            margin-bottom:30px;
        ">
            Real-time overview of sales, customers and product performance
        </div>
        """,
        unsafe_allow_html=True
    )

    # ====================================
    # KPI SECTION
    # ====================================

    total_customers = (
        retail_filtered["Customer ID"]
        .nunique()
    )

    total_revenue = (
        retail_filtered["total_price"]
        .sum()
    )

    total_orders = (
        retail_filtered["Invoice"]
        .nunique()
    )

    avg_order = (
        total_revenue /
        max(total_orders, 1)
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">👥 Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">💰 Revenue</div>
            <div class="kpi-value">₹{total_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">📦 Orders</div>
            <div class="kpi-value">{total_orders:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="custom-kpi">
            <div class="kpi-label">🛒 Avg Order</div>
            <div class="kpi-value">₹{avg_order:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)


    section_space(35)

    # ====================================
    # REVENUE TREND + DONUT
    # ====================================

    left, right = st.columns([2, 1])

    with left:

        monthly_revenue = (
            retail_filtered
            .groupby(
                "InvoiceMonth",
                as_index=False
            )
            .agg(
                Revenue=(
                    "total_price",
                    "sum"
                )
            )
        )

        fig = px.area(
            monthly_revenue,
            x="InvoiceMonth",
            y="Revenue",
            title="📈 Monthly Revenue Trend"
        )

        fig.update_traces(
            line_color="#2563EB"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue (₹)",
            height=480,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=40
            )
        )

        fig = style_plotly(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        top_products_share = (
            retail_filtered
            .groupby(
                "Description"
            )["total_price"]
            .sum()
            .nlargest(5)
            .reset_index()
        )

        fig = px.pie(
            top_products_share,
            names="Description",
            values="total_price",
            hole=0.45,
            title="🥧 Revenue Share",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            legend=dict(
                orientation="v",
                x=0.98,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
            margin=dict(
                l=60,
                r=30,
                t=70,
                b=60
            )
        )

        fig = style_plotly(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    section_space(35)
    # ====================================
    # THREE CHARTS SECTION
    # ====================================

    col1, col2, col3 = st.columns(3)

    # --------------------------------
    # TOP PRODUCTS
    # --------------------------------

    with col1:

        top_products = (
            retail_filtered
            .groupby(
                "Description"
            )["total_price"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="total_price",
            y="Description",
            orientation="h",
            title="🔥 Top Products",
            color="total_price",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            xaxis_title="Revenue (₹)",
            yaxis_title="Products",
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=40
            ),
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        fig = style_plotly(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------
    # MONTHLY ORDERS
    # --------------------------------

    with col2:

        monthly_orders = (
            retail_filtered
            .groupby(
                "InvoiceMonth"
            )["Invoice"]
            .nunique()
            .reset_index()
        )
    
        fig = px.bar(
            monthly_orders,
            x="InvoiceMonth",
            y="Invoice",
            title="📦 Monthly Orders",
            color="Invoice",
            color_continuous_scale="Purples"
        )
    
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Orders",
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=40
            )
        )
    
        fig = style_plotly(fig)
    
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    
    # --------------------------------
    # CUSTOMER SPEND DISTRIBUTION
    # --------------------------------
    
    with col3:
    
        customer_spend = (
            retail_filtered
            .groupby(
                "Customer ID"
            )["total_price"]
            .sum()
            .reset_index()
        )
    
        fig = px.histogram(
            customer_spend,
            x="total_price",
            nbins=30,
            title="👥 Customer Spend Distribution",
            color_discrete_sequence=[
                "#7C3AED"
            ]
        )
    
        fig.update_layout(
            xaxis_title="Customer Spend (₹)",
            yaxis_title="Customers",
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=40
            )
        )
    
        fig = style_plotly(fig)
    
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    
    section_space(35)
    # ====================================
    # BOTTOM KPI ROW
    # ====================================

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "📦 Quantity Sold",
        f"{retail_filtered['Quantity'].sum():,.0f}"
    )

    b2.metric(
        "🏷 Unique Products",
        retail_filtered[
            "StockCode"
        ].nunique()
    )

    b3.metric(
        "🧾 Transactions",
        retail_filtered[
            "Invoice"
        ].nunique()
    )

    b4.metric(
        "💵 Avg Product Price",
        f"₹{retail_filtered['Price'].mean():.2f}"
    )