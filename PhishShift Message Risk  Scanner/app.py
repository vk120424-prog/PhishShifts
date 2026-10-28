"""
PhishShift - Message Risk Scanner
Interactive Cybersecurity Web Application built with Streamlit.
Analyzes email and SMS messages for phishing, deceptive links, and social engineering patterns.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from message_analyzer import MessageAnalyzer
from train_model import train_pipeline, DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PhishShift – Message Risk Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Dark Navy Background and Global Styles */
    .stApp {
        background-color: #0a0f1d;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Brand Card */
    .brand-header {
        background: linear-gradient(135deg, #0d1527 0%, #131d38 100%);
        border: 1px solid #1e293b;
        border-left: 4px solid #00f0ff;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 240, 255, 0.08);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-highlight {
        color: #00f0ff;
        text-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Result Cards */
    .risk-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
    }
    .risk-card-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);
    }
    .risk-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }

    /* Metric Badges */
    .risk-badge {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }

    /* Recommendation Box */
    .rec-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 14px;
        color: #f1f5f9;
        font-size: 0.95rem;
    }

    /* Indicator Pill */
    .indicator-pill {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Sidebar info box */
    .sidebar-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Preset demonstration messages for instant testing
EXAMPLE_MESSAGES = {
    "Select an example to load...": "",
    "AT-01: Legitimate Password Reset (GitHub)": (
        "Your GitHub password was recently changed on Oct 27, 2026. "
        "If you did this, no further action is required. If not, secure your account at github.com."
    ),
    "Legitimate: Team Meeting Reminder": (
        "Hi team, reminder that our weekly engineering sync is tomorrow at 10:00 AM in Conference Room B. "
        "Please update your Jira tickets beforehand."
    ),
    "Legitimate: Amazon Order Shipped": (
        "Your Amazon order #112-8829103 has shipped via UPS. "
        "You can track your package directly inside the official Amazon application."
    ),
    "Phishing: Urgent Bank Account Suspension (Chase)": (
        "URGENT: Your Chase banking account has been temporarily locked due to suspicious transactions! "
        "Verify your password and SSN immediately to restore access: http://chase-security-verify.click/login"
    ),
    "Smishing: USPS Incomplete Address (bit.ly Shortener)": (
        "USPS Notice: Your parcel #US940291 is on hold due to missing street address. "
        "Update delivery details within 24 hours or package will be returned: bit.ly/usps-track-99"
    ),
    "Credential Harvesting: Microsoft 365 Expiration": (
        "FINAL WARNING: Your Microsoft 365 password expires today! "
        "Keep your current password by verifying credentials now at: http://micros0ft-password-renewal.top"
    ),
    "Scam: International Lottery & Wire Transfer (IP Host)": (
        "Dear Customer, you have won $1,000,000 in the International Mobile Lottery! "
        "Send your full name, bank account number, and wire $250 fee to claim: http://192.168.1.55/claim"
    ),
    "Scam: Crypto Giveaway Doubling Scheme": (
        "Exclusive Crypto Opportunity: Elon Musk is doubling all Bitcoin deposits for the next 30 minutes! "
        "Send 0.1 BTC to receive 0.2 BTC instantly: http://192.168.10.88/btc-giveaway"
    )
}


@st.cache_resource
def get_analyzer() -> MessageAnalyzer:
    """Initializes and caches the MessageAnalyzer instance."""
    return MessageAnalyzer()


def main():
    analyzer = get_analyzer()

    # Session State Initialization
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ System Status")
        
        # Model Status badge
        if analyzer.is_model_loaded:
            st.success("🟢 ML Model: Active & Loaded")
            if analyzer.metrics:
                st.caption(f"Trained: {analyzer.metrics.get('trained_at', 'Recently')}")
                st.caption(f"Test Accuracy: {analyzer.metrics.get('accuracy', 0.0):.1%}")
        else:
            st.warning("🟡 ML Model: Not Loaded")
            st.info("System is operating using transparent rule-based heuristics.")
            if st.button("⚡ Train Model Now", use_container_width=True):
                with st.spinner("Training machine learning pipeline..."):
                    try:
                        metrics = train_pipeline()
                        analyzer.load_model()
                        st.success("Model trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        st.markdown("---")
        st.markdown("### 📊 Risk Scoring Scale")
        st.markdown(
            """
            - 🟢 **LOW RISK (0 – 34)**: No major suspicious patterns found.
            - 🟡 **MEDIUM RISK (35 – 69)**: Elevated urgency or link obfuscation.
            - 🔴 **HIGH RISK (70 – 100)**: Strong indicators of phishing/theft.
            
            *Score is a heuristic risk indicator, not a calibrated attack probability.*
            """
        )

        st.markdown("---")
        st.markdown("### 🔒 Cybersecurity Safety Tips")
        st.markdown(
            """
            1. **Never share passwords** or 2FA codes via email/SMS.
            2. **Inspect sender domain** carefully for subtle typos.
            3. **Avoid clicking shortened links** from unexpected senders.
            4. **Verify independently** using official bank or service apps.
            """
        )

        if st.session_state.session_history:
            st.markdown("---")
            if st.button("🗑️ Clear Session History", use_container_width=True):
                st.session_state.session_history = []
                st.session_state.current_analysis = None
                st.success("Session history cleared.")
                st.rerun()

    # Main Header
    st.markdown(
        """
        <div class="brand-header">
            <h1 class="brand-title">
                🛡️ Phish<span class="brand-highlight">Shift</span>
            </h1>
            <p class="brand-subtitle">
                Advanced Message Risk Scanner — Inspect email and SMS communications for phishing intent, 
                urgency coercion, deceptive links, and shortened URLs safely offline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Primary Navigation Tabs
    tab_scan, tab_metrics, tab_history, tab_about = st.tabs([
        "🔍 Message Risk Scanner",
        "📈 Model Performance",
        "📜 Session History",
        "ℹ️ About & Threat Heuristics"
    ])

    # -------------------------------------------------------------
    # TAB 1: SCANNER
    # -------------------------------------------------------------
    with tab_scan:
        st.markdown("#### Submit Message for Inspection")

        # Example selector
        selected_example = st.selectbox(
            "📋 Quick Test Samples (Click to populate input):",
            list(EXAMPLE_MESSAGES.keys()),
            key="example_selector"
        )
        if selected_example and EXAMPLE_MESSAGES[selected_example]:
            st.session_state.input_text = EXAMPLE_MESSAGES[selected_example]

        # Message Input
        user_message = st.text_area(
            "Paste Email or SMS Message Content:",
            value=st.session_state.input_text,
            height=150,
            placeholder="e.g. URGENT: Your account has been suspended. Click here to verify credentials...",
            help="Your submitted text is evaluated strictly in-memory and is never stored permanently."
        )
        st.session_state.input_text = user_message

        # Word & Char counts
        char_count = len(user_message)
        word_count = len(user_message.split()) if user_message else 0
        st.caption(f"Characters: {char_count} | Words: {word_count}")

        # Action Buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            analyze_clicked = st.button("⚡ Analyze Message", type="primary", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("🧹 Clear Input", use_container_width=True)
        with col_btn3:
            random_clicked = st.button("🎲 Random Sample", use_container_width=True)

        if clear_clicked:
            st.session_state.input_text = ""
            st.session_state.current_analysis = None
            st.rerun()

        if random_clicked:
            keys = [k for k in EXAMPLE_MESSAGES.keys() if EXAMPLE_MESSAGES[k]]
            chosen = np.random.choice(keys)
            st.session_state.input_text = EXAMPLE_MESSAGES[chosen]
            st.rerun()

        if analyze_clicked:
            if not user_message.strip():
                st.warning("⚠️ Please enter or paste message text to analyze.")
            else:
                with st.spinner("Analyzing message text, heuristics, and URLs..."):
                    result = analyzer.analyze(user_message)
                    st.session_state.current_analysis = result

                    # Record to session history
                    st.session_state.session_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "preview": (user_message[:55] + "...") if len(user_message) > 55 else user_message,
                        "risk_score": result["risk_score"],
                        "risk_category": result["risk_category"],
                        "predicted_category": result["predicted_category"],
                        "url_count": result["url_analysis"]["url_count"],
                        "flag_count": len(result["indicators"]) + len(result["url_analysis"]["url_findings"])
                    })

        # Display Analysis Results
        if st.session_state.current_analysis:
            res = st.session_state.current_analysis

            st.markdown("---")
            st.markdown("### 📋 Risk Assessment Results")

            # Determine styling by risk category
            cat = res["risk_category"]
            score = res["risk_score"]
            if cat == "HIGH RISK":
                card_class = "risk-card-high"
                badge_class = "badge-high"
                score_color = "#ef4444"
            elif cat == "MEDIUM RISK":
                card_class = "risk-card-med"
                badge_class = "badge-med"
                score_color = "#f59e0b"
            else:
                card_class = "risk-card-low"
                badge_class = "badge-low"
                score_color = "#10b981"

            # Result Card HTML
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span class="risk-badge {badge_class}">{cat}</span>
                            <span style="margin-left: 14px; font-size: 1.1rem; color: #94a3b8;">
                                Classification: <strong style="color: #ffffff;">{res['predicted_category']}</strong>
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.9rem; color: #94a3b8;">Heuristic Risk Score:</span>
                            <div style="font-size: 2.2rem; font-weight: 800; color: {score_color}; line-height: 1.1;">
                                {score} <span style="font-size: 1.1rem; color: #64748b;">/ 100</span>
                            </div>
                        </div>
                    </div>
                    <div class="rec-box">
                        <strong>🛡️ Safety Recommendation:</strong> {res['recommendation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Heuristic Progress Bar
            st.progress(score / 100.0)

            # Details Grid
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("#### 🎯 Machine Learning & Heuristics Breakdown")
                ml_res = res["ml_result"]
                if ml_res.get("available"):
                    st.markdown(f"- **ML Model Prediction:** `{ml_res['label'].capitalize()}`")
                    st.markdown(f"- **ML Confidence:** `{ml_res['confidence_percent']}%`")
                    st.markdown(f"- **Estimated Phishing Probability:** `{ml_res.get('phishing_probability', 0.0):.1%}`")
                else:
                    st.markdown("- **ML Model Status:** `Not loaded (Heuristic evaluation active)`")

                st.markdown(f"- **Keyword/Urgency Penalty:** `{res['heuristic_breakdown']['keyword_points']} pts`")
                st.markdown(f"- **URL Risk Contribution:** `{res['heuristic_breakdown']['url_points']} pts`")
                st.markdown(f"- **Formatting Anomalies Penalty:** `{res['heuristic_breakdown']['formatting_points']} pts`")

                st.markdown("#### 💡 Why Was This Message Flagged?")
                for r in res["reasons"]:
                    st.markdown(f"- {r}")

            with col_right:
                st.markdown("#### 🔗 Offline URL Inspection Findings")
                url_data = res["url_analysis"]
                st.markdown(f"- **URLs Detected:** `{url_data['url_count']}`")
                st.markdown(f"- **URL Shorteners Used:** `{'Yes (High Concealment)' if url_data['has_shorteners'] else 'No'}`")
                st.markdown(f"- **IP Address Hosts Detected:** `{'Yes (Suspicious)' if url_data['has_ip_urls'] else 'No'}`")
                st.markdown(f"- **Brand Imitation Identified:** `{'Yes (Deceptive Domain)' if url_data['has_brand_imitation'] else 'No'}`")

                if url_data["urls"]:
                    with st.expander("Extracted URLs (Inspected Safely Offline)"):
                        for u in url_data["urls"]:
                            st.code(u, language="text")

                # URL Findings list
                if url_data["url_findings"]:
                    st.markdown("**Specific URL Findings:**")
                    for uf in url_data["url_findings"]:
                        st.markdown(f"- ⚠️ {uf}")

                # URL Static Inspection Disclaimer
                st.info(f"ℹ️ {url_data['disclaimer']}")

            # Suspicious Indicators Table
            if res["indicators"]:
                st.markdown("#### 🚨 Detected Suspicious Pattern Indicators")
                ind_df = pd.DataFrame(res["indicators"])
                ind_df = ind_df.rename(columns={
                    "category": "Indicator Category",
                    "term": "Detected Snippet",
                    "explanation": "Threat Significance",
                    "points": "Risk Points"
                })
                st.dataframe(ind_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 2: MODEL PERFORMANCE
    # -------------------------------------------------------------
    with tab_metrics:
        st.markdown("### 📈 Machine Learning Model Performance")

        if analyzer.metrics:
            m = analyzer.metrics
            
            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Test Accuracy", f"{m['accuracy']:.1%}")
            with c2:
                st.metric("Precision (Phishing)", f"{m['precision']:.1%}")
            with c3:
                st.metric("Recall (Phishing)", f"{m['recall']:.1%}")
            with c4:
                st.metric("F1-Score (Phishing)", f"{m['f1_score']:.1%}")

            st.markdown("---")
            col_m1, col_m2 = st.columns([1, 1])

            with col_m1:
                st.markdown("#### Confusion Matrix (Test Split)")
                cm = np.array(m["confusion_matrix"])
                
                # Matplotlib visualization with dark styling
                fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="#0d1527")
                ax.set_facecolor("#0d1527")
                
                cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
                fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

                labels = ["Legitimate", "Phishing"]
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(labels, color="#e2e8f0", fontsize=10)
                ax.set_yticklabels(labels, color="#e2e8f0", fontsize=10)
                ax.set_xlabel("Predicted Label", color="#94a3b8", labelpad=8)
                ax.set_ylabel("True Label", color="#94a3b8", labelpad=8)

                for (i, j), val in np.ndenumerate(cm):
                    ax.text(
                        j, i, f"{val}",
                        ha="center", va="center",
                        color="#ffffff" if val > (cm.max() / 2) else "#00f0ff",
                        fontweight="bold", fontsize=14
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_m2:
                st.markdown("#### Detailed Classification Report")
                st.code(m.get("classification_report_text", "No report text available"), language="text")

                st.markdown("#### Training Specifications")
                st.markdown(f"- **Classifier:** `{m.get('model_type', 'Logistic Regression')}`")
                st.markdown(f"- **Total Dataset Samples:** `{m.get('total_samples', 0)}`")
                st.markdown(f"- **Training Set Size:** `{m.get('train_samples', 0)}`")
                st.markdown(f"- **Testing Set Size:** `{m.get('test_samples', 0)}`")
                st.markdown(f"- **Feature Extraction:** `TF-IDF (unigrams & bigrams, sublinear TF)`")

            st.markdown("---")
            st.markdown("#### 🔬 Dataset Provenance & Real-World Limitations")
            st.info(
                f"**Provenance:** {m.get('provenance_note', 'Trained on demonstrative curated dataset.')}\n\n"
                "**Critical Limitation Notice:** Machine learning models evaluate vocabulary and structural patterns "
                "observed in training data. Attackers frequently adapt with novel phishing templates, obfuscated Unicode, "
                "or legitimate compromised accounts. A low-risk classification does not guarantee complete absence of risk."
            )
        else:
            st.warning("⚠️ No trained model metrics found. Please train the model using `train_model.py` or the sidebar button.")
            if st.button("🚀 Train Model Now", key="train_btn_metrics"):
                with st.spinner("Training model..."):
                    train_pipeline()
                    analyzer.load_model()
                    st.success("Model trained successfully!")
                    st.rerun()

    # -------------------------------------------------------------
    # TAB 3: SESSION HISTORY
    # -------------------------------------------------------------
    with tab_history:
        st.markdown("### 📜 Current Session Message History")
        if not st.session_state.session_history:
            st.info("No messages have been analyzed in this session yet. Submit a message in the scanner tab to see it logged here.")
        else:
            hist_df = pd.DataFrame(st.session_state.session_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Session History (CSV)",
                data=csv_data,
                file_name="phishshift_session_history.csv",
                mime="text/csv",
                use_container_width=True
            )

    # -------------------------------------------------------------
    # TAB 4: ABOUT & THREAT HEURISTICS
    # -------------------------------------------------------------
    with tab_about:
        st.markdown("### ℹ️ About PhishShift – Message Risk Scanner")
        st.markdown(
            """
            **PhishShift** is an educational cybersecurity tool designed to analyze suspicious text communications
            (emails, SMS, chat messages) and explain risk indicators without exposing users to harm.

            #### Key Defensive Features
            1. **Transparent Rule-Based Heuristics**: Detects social engineering hallmarks including artificial urgency,
               unauthorized access claims, password harvesting, and financial scam lures.
            2. **Safe, Offline URL Analysis**: Inspects extracted URLs for URL shorteners, raw IP address hosts,
               typosquatting/brand imitation, and excessive subdomains **without executing HTTP requests or resolving DNS**.
            3. **Machine Learning Text Classification**: Employs TF-IDF feature extraction combined with balanced Logistic
               Regression to classify message intent based on linguistic patterns.
            4. **Explainable AI (XAI)**: Demystifies cybersecurity by explaining precisely which phrases, domains, or
               structural markers contributed to the heuristic risk score.

            #### Privacy & Educational Notice
            - Messages submitted are evaluated purely in-memory during your session and are **never logged or persisted to disk**.
            - This application is developed for educational demonstrations and awareness; it is **not a substitute for enterprise security suites**.
            """
        )


if __name__ == "__main__":
    main()
