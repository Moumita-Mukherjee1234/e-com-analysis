import streamlit as st
import pandas as pd
import plotly.express as px
from utils.style import style_plotly
import streamlit.components.v1 as components


def show_prediction(rf_model):

    # ==========================
    # GLOBAL STYLE
    # ==========================
    st.markdown("""
    <style>

    .page-title {
        font-size: 38px;
        font-weight: 800;
        color: #111111;
        text-align: left;
        margin-bottom: 5px;
    }

    .sub-text {
        text-align: left;
        color: #666;
        margin-bottom: 30px;
        font-size: 15px;
    }

    /* KPI INPUT CARDS */
    .kpi-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #E7DFFF;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        text-align: center;
    }

    .kpi-title {
        font-size: 14px;
        font-weight: 600;
        color: #5E51E0;
        margin-bottom: 8px;
    }

    /* PREDICT BUTTON */
    div.stButton > button {
        width: 100%;
        padding: 14px;
        background: #E9E3FF;
        color: #2A2A2A;
        font-weight: 700;
        border-radius: 12px;
        border: 1px solid #CFC3FF;
    }

    div.stButton > button:hover {
        background: #DCD2FF;
        border: 1px solid #B8A7FF;
    }

    /* RESULT CARDS */
    .result-card {
        background: white;
        border: 1px solid #E7DFFF;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    .result-title {
        font-size: 14px;
        color: #666;
    }

    .result-value {
        font-size: 22px;
        font-weight: 700;
        color: #2A2A2A;
        margin-top: 6px;
    }

    .good {
        color: #2E7D32;
    }

    .bad {
        color: #C62828;
    }
            

    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>

    .confidence-card{
        background:white;
        border:1px solid #E7DFFF;
        border-radius:18px;
        padding:24px;
        display:flex;
        justify-content:space-between;
        align-items:center;
        box-shadow:0 4px 12px rgba(0,0,0,0.05);
    }

    .conf-left{
        flex:1;
    }

    .conf-title{
        font-size:14px;
        font-weight:600;
        color:#6F61FF;
        margin-bottom:10px;
    }

    .conf-value{
        font-size:34px;
        font-weight:700;
        color:#2A2A2A;
        margin-bottom:8px;
    }

    .conf-subtext{
        font-size:13px;
        color:#7A7A7A;
    }

    .ring{
        width:100px;
        height:100px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
    }

    .ring-inner{
        width:72px;
        height:72px;
        background:white;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:16px;
        font-weight:700;
        color:#2A2A2A;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # ==========================
    # HEADER
    # ==========================
    st.markdown('<div class="page-title">🎯 PURCHASE PREDICTION</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sub-text">Predict whether a customer will purchase using RFM metrics</div>',
        unsafe_allow_html=True
    )

    # ==========================
    # INPUT SECTION - SAAS CARDS (FINAL CLEAN VERSION)
    # ==========================

    st.markdown("""
    <style>

    .input-card {
        background: white;
        border: 1px solid #E7DFFF;
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        transition: all 0.25s ease;
        min-height: 95px;
    }

    .input-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 35px rgba(124,58,237,0.15);
    }
    
    /* ==========================
       HEADER INSIDE CARD
    ========================== */
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    
    .icon {
        font-size: 20px;
    }
    
    .title {
        font-size: 15px;
        font-weight: 800;
        color: #5B5AF7;
    }
    
    .helper {
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 8px;
    }
    
    /* ==========================
       NUMBER INPUT STYLING
    ========================== */
    
    .stNumberInput > div > div > input {
        background: #FFFFFF !important;
        border: 1px solid #E7DFFF !important;
        border-radius: 12px !important;
        padding: 8px 10px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        transition: all 0.25s ease;
    }
    
    /* Focus glow */
    .stNumberInput > div > div > input:focus {
        border: 1px solid #7C3AED !important;
        box-shadow: 0 0 0 4px rgba(124,58,237,0.15) !important;
        outline: none !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)

    # ==========================
    # RECENCY CARD
    # ==========================
    with col1:
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">📅</div>
                <div class="title">Recency</div>
            </div>
            <div class="helper">Days since last purchase</div>
        </div>
        <br>
        """, unsafe_allow_html=True)
    
        r = st.number_input(
            label="Recency input",
            min_value=0,
            value=30,
            step=1,
            label_visibility="collapsed",
            key="recency"
        )
    
    # ==========================
    # FREQUENCY CARD
    # ==========================
    with col2:
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">🔁</div>
                <div class="title">Frequency</div>
            </div>
            <div class="helper">Number of purchases</div>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        f = st.number_input(
            label="Frequency input",
            min_value=0,
            value=2,
            step=1,
            label_visibility="collapsed",
            key="frequency"
        )
    
    # ==========================
    # MONETARY CARD
    # ==========================
    with col3:
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">💰</div>
                <div class="title">Monetary</div>
            </div>
            <div class="helper">Total spend (₹)</div>
        </div>
        <br>
        """, unsafe_allow_html=True)
    
        m = st.number_input(
            label="Monetary input",
            min_value=0,
            value=500,
            step=50,
            label_visibility="collapsed",
            key="monetary"
        )
    # ==========================
    # PREDICT BUTTON
    # ==========================
    predict = st.button("🔮 Predict Purchase Behavior")

    # ==========================
    # PREDICTION LOGIC
    # ==========================
    if predict:

        try:

            input_df = pd.DataFrame({
                "recency_days": [r],
                "frequency": [f],
                "monetary": [m]
            })

            prediction = rf_model.predict(input_df)[0]
            confidence = rf_model.predict_proba(input_df).max()


            # ==========================
            # RESULT CARDS
            # ==========================
            col1, col2 = st.columns(2)
    
            # --------------------------
            # PREDICTION CARD
            # --------------------------
            with col1:
    
                if prediction == 1:
    
                    st.markdown("""
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-icon">🛡️</span>
                            <div class="result-content">
                                <div class="result-title good">Prediction</div>
                                <div class="result-value good">Will Buy</div>
                                <div class="result-subtitle">
                                    Customer is likely to make another purchase
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
                else:
    
                    st.markdown("""
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-icon">⚠️</span>
                            <div class="result-content">
                                <div class="result-title bad">Prediction</div>
                                <div class="result-value bad">Will Not Buy</div>
                                <div class="result-subtitle">
                                    Customer is unlikely to make another purchase
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
            # --------------------------
            # CONFIDENCE CARD
            # --------------------------
                

            with col2:
                confidence_pct = confidence * 100

                components.html(
                    f"""
                    <html>
                    <head>
                    <style>
                
                    body {{
                        margin:0;
                        padding:0;
                        font-family:Arial, sans-serif;
                    }}
                
                    .confidence-card {{
                        background:white;
                        border:1px solid #E7DFFF;
                        border-radius:18px;
                        padding:22px;
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        box-sizing:border-box;
                        
                    }}
                
                    .conf-left {{
                        flex:1;
                    }}
                
                    .conf-title {{
                        font-size:18px;
                        font-weight:600;
                        color:#6F61FF;
                        margin-bottom:10px;
                    }}
                
                    .conf-value {{
                        font-size:36px;
                        font-weight:800;
                        color:#2A2A2A;
                        margin-bottom:8px;
                    }}
                
                    .conf-subtext {{
                        font-size:15px;
                        color:#7A7A7A;
                    }}
                
                    .ring {{
                        width:120px !important;
                        height:120px !important;
                        border-radius:50%;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                    }}
                
                    .ring-inner {{
                        width:80px !important;
                        height:80px !important;
                        border-radius:50%;
                        background:white;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:16px;
                        font-weight:700;
                        color:#e5b9f3;
                    }}
                
                    </style>
                    </head>

                    <body>

                    <div class="confidence-card">

                        <div class="conf-left">
                            <div class="conf-title">
                                🎯 Model Confidence
                            </div>
                
                            <div class="conf-value">
                                {confidence_pct:.2f}%
                            </div>
                
                            <div class="conf-subtext">
                                Model certainty for this prediction
                            </div>
                        </div>
                
                        <div
                            class="ring"
                            style="
                                background:
                                conic-gradient(
                                    #8A6ED7 0% {confidence_pct}%,
                                    #e5b9f3 {confidence_pct}% 100%
                                );
                            "
                        >
                            <div class="ring-inner">
                                {confidence_pct:.0f}%
                            </div>
                        </div>
                
                    </div>
                
                    </body>
                    </html>
                    """,
                    height=170 ,
                )

            # ==========================
            # FEATURE IMPORTANCE + RECOMMENDATIONS
            # ==========================

            st.markdown("<br>", unsafe_allow_html=True)

            col3, col4 = st.columns(2)

            # --------------------------------
            # FEATURE IMPORTANCE
            # --------------------------------

            with col3:

                st.markdown("### 📊 Feature Importance")

                feature_importance = rf_model.feature_importances_

                importance_df = pd.DataFrame({
                    "Feature": ["Recency", "Frequency", "Monetary"],
                    "Importance": feature_importance
                })
            
                importance_df["Importance"] = (
                    importance_df["Importance"] /
                    importance_df["Importance"].sum()
                ) * 100
            
                importance_df = importance_df.sort_values(
                    "Importance",
                    ascending=True
                )
            
                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    text=importance_df["Importance"].round(1),
                    title="evaluated features"
                )
            


                fig = style_plotly(fig)
                fig.update_layout(

                    height=320,
                    margin=dict(l=10, r=20, t=20, b=10),
                    xaxis_title="Importance (%)",
                    yaxis_title="",
                    showlegend=False,
                    title_x=0,
                    title_y=0.99,
                    
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            
            # --------------------------------
            # RECOMMENDATIONS
            # --------------------------------
        
            with col4:
                st.markdown("### 💡 Recommended Actions")
                st.markdown(" ")

                recommendations = []

                if r > 90:
                    recommendations.append(
                        "📧 Launch an aggressive win-back campaign."
                    )

                elif r > 60:
                    recommendations.append(
                        "📨 Send personalized re-engagement emails."
                    )

                elif r < 15:
                    recommendations.append(
                        "🔥 Customer is highly active. Maintain engagement."
                    )

                # ==========================
                # FREQUENCY RULES
                # ==========================

                if f >= 10:
                    recommendations.append(
                        "🏆 Enroll customer in VIP loyalty program."
                    )

                elif f >= 5:
                    recommendations.append(
                        "🎁 Offer loyalty rewards for repeat purchases."
                    )

                elif f <= 2:
                    recommendations.append(
                        "🛍️ Encourage repeat purchases with coupons."
                    )

                # ==========================
                # MONETARY RULES
                # ==========================
                
                if m >= 3000:
                    recommendations.append(
                        "👑 Promote premium memberships and exclusive benefits."
                    )
                
                elif m >= 1000:
                    recommendations.append(
                        "💰 Offer premium product recommendations."
                    )
                
                elif m < 500:
                    recommendations.append(
                        "🏷️ Provide discount bundles to increase spending."
                    )
                
                # ==========================
                # COMBINED RFM RULES
                # ==========================
                
                if r < 30 and f >= 5:
                    recommendations.append(
                        "🛒 Recommend complementary products through cross-selling."
                    )
                
                if r < 30 and m >= 1000:
                    recommendations.append(
                        "⭐ Offer early access to new products."
                    )
                
                if r > 60 and m >= 1000:
                    recommendations.append(
                        "☎️ High-value customer at risk. Trigger retention campaign."
                    )
                
                if r > 90 and f <= 2:
                    recommendations.append(
                        "⚠️ Customer shows strong churn signals."
                    )
                
                if f >= 8 and m >= 2000:
                    recommendations.append(
                        "💎 Consider this customer for VIP segmentation."
                    )
                
                # ==========================
                # PREDICTION BASED RULES
                # ==========================
                
                if prediction == 1:
                
                    recommendations.append(
                        "🚀 Upsell premium products and memberships."
                    )
                
                    recommendations.append(
                        "🎯 Use personalized recommendations to increase basket size."
                    )
                
                else:
                
                    recommendations.append(
                        "📢 Run targeted promotional campaigns."
                    )
                
                    recommendations.append(
                        "🎁 Offer limited-time incentives to reactivate interest."
                    )
                
                # ==========================
                # FALLBACK
                # ==========================
                
                if len(recommendations) < 3:
                
                    recommendations.extend([
                        "📈 Monitor customer activity regularly.",
                        "💡 Improve personalization efforts.",
                        "🤝 Maintain consistent customer engagement."
                    ])
                
                # Remove duplicates
                recommendations = list(dict.fromkeys(recommendations))
                            
                recommendation_html = ""
                for rec in recommendations:

                    recommendation_html += f"""
                    <div style="
                        margin-bottom:16px;
                        font-size:15px;
                        font-weight:500;
                    ">
                        {rec}
                    </div>
                    """
            
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(
                            135deg,
                            #C4B5FD,
                            #A78BFA
                        );
                        padding:24px;
                        border-radius:18px;
                        color:black;
                        font-size:15px;
                        line-height:1.6;
                        box-shadow:0 10px 25px rgba(
                            124,
                            58,
                            237,
                            0.18
                        );
                        min-height:340px;
                    ">
            
            
                    <div style="
                        font-size:14px;
                        margin-bottom:22px;
                        opacity:0.9;
                    ">
                        Suggested strategies based on
                        customer behavior
                    </div>
            
                    {recommendation_html}
            
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
    
            st.error("Prediction failed.")
            st.exception(e)