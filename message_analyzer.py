"""
PhishShift - Message Risk Analyzer Module
Combines ML text classification with explainable cybersecurity heuristics
and offline URL analysis to assess phishing and scam risk.
"""

import os
import re
from typing import Dict, List, Any, Optional, Tuple
import joblib

from url_analyzer import analyze_urls

# Default paths for saved model artifacts
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "model.joblib")
DEFAULT_VECTORIZER_PATH = os.path.join(DEFAULT_MODELS_DIR, "vectorizer.joblib")
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_MODELS_DIR, "metrics.joblib")

# 1. Urgent & Coercive Language Patterns
URGENT_PATTERNS = [
    (r"\b(?:urgent|immediately|immediate action required|act now|within \d+ hours?|today only|expires? today|final notice|final warning|critical alert)\b",
     "High urgency or artificial panic phrase designed to force quick action", 15),
    (r"\b(?:unauthorized (?:activity|access|login|transaction|withdrawal)|suspicious (?:activity|transaction)|account (?:locked|suspended|restricted|terminated|blocked|on hold))\b",
     "Claims account compromise, security lock, or suspension", 20),
    (r"\b(?:click (?:here|the link|below|immediately)|log in immediately|restore access now|cancel payment now|verify now|unblock)\b",
     "Direct call-to-action urging immediate clicking or login", 15)
]

# 2. Credential & Sensitive Data Requests
CREDENTIAL_PATTERNS = [
    (r"\b(?:verify|confirm|update|enter|submit)\b.{0,35}\b(?:password|passcode|pin|ssn|social security|credentials|username|login details|cvv|security code)\b",
     "Explicit request for passwords, PINs, SSN, or confidential credentials", 35),
    (r"\b(?:confirm your identity|identity verification|verify your account|update (?:billing|card|banking) details)\b",
     "Request to verify identity or re-submit financial details", 20)
]

# 3. Financial & Scam Incentives
FINANCIAL_PATTERNS = [
    (r"\b(?:won|winner|lottery|prize|reward|\$\d+[\d,]* (?:gift card|cash|bonus|refund)|congratulations.{0,30}(?:selected|won))\b",
     "Enticing claim of winning lottery, cash prize, or unexpected reward", 25),
    (r"\b(?:wire transfer|crypto|bitcoin|btc|gift card|western union|moneygram|customs fee|unauthorized wire)\b",
     "References untraceable payment methods, gift cards, or crypto transfers", 25),
    (r"\b(?:tax refund|direct deposit routing|internal revenue service|irs)\b",
     "References tax refund or direct deposit authority", 20)
]


class MessageAnalyzer:
    """
    Main analyzer combining Machine Learning text classification with
    rule-based indicator detection and safe offline URL analysis.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
        metrics_path: str = DEFAULT_METRICS_PATH
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.metrics_path = metrics_path
        self.model = None
        self.vectorizer = None
        self.metrics = None
        self.is_model_loaded = False
        self.load_model()

    def load_model(self) -> bool:
        """
        Attempts to load the trained model and vectorizer.
        Returns True if successful, False otherwise without raising errors.
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_model_loaded = True
                if os.path.exists(self.metrics_path):
                    self.metrics = joblib.load(self.metrics_path)
                return True
        except Exception as e:
            print(f"[*] Notice: Model could not be loaded ({e}). Operating in heuristic-only mode.")
            self.model = None
            self.vectorizer = None
            self.metrics = None
            self.is_model_loaded = False
        return False

    def detect_formatting_anomalies(self, text: str) -> Dict[str, Any]:
        """
        Inspects text for excessive punctuation, capitalization, or formatting anomalies.
        """
        flags = []
        points = 0

        # Excessive exclamation marks (3 or more)
        exclamation_count = text.count("!")
        if exclamation_count >= 3:
            flags.append(f"Excessive exclamation marks ({exclamation_count} detected)")
            points += 10

        # Excessive question marks
        question_count = text.count("?")
        if question_count >= 3:
            flags.append(f"Multiple consecutive question marks ({question_count} detected)")
            points += 5

        # All-caps words analysis (words of length >= 4)
        words = re.findall(r"\b[A-Za-z]{4,}\b", text)
        if words:
            caps_words = [w for w in words if w.isupper()]
            caps_ratio = len(caps_words) / len(words)
            if len(caps_words) >= 2 or caps_ratio >= 0.35:
                flags.append(f"Excessive uppercase/capitalized words ({len(caps_words)} words, e.g. {caps_words[:3]})")
                points += 15

        return {
            "flags": flags,
            "points": min(points, 25),
            "exclamation_count": exclamation_count,
            "caps_words_count": len([w for w in words if w.isupper()]) if words else 0
        }

    def detect_keyword_indicators(self, text: str) -> Dict[str, Any]:
        """
        Scans message content for cybersecurity risk patterns and categorizes them.
        """
        detected_indicators = []
        reasons = []
        total_points = 0

        # Urgent language checks
        for pattern, reason, pts in URGENT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                snippet = match.group(0)
                detected_indicators.append({
                    "category": "Urgency & Panic",
                    "term": snippet,
                    "explanation": reason,
                    "points": pts
                })
                reasons.append(f"Urgent phrasing: '{snippet}' ({reason})")
                total_points += pts

        # Credential checks
        for pattern, reason, pts in CREDENTIAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                snippet = match.group(0)
                detected_indicators.append({
                    "category": "Credential Harvesting",
                    "term": snippet,
                    "explanation": reason,
                    "points": pts
                })
                reasons.append(f"Credential solicitation: '{snippet}' ({reason})")
                total_points += pts

        # Financial checks
        for pattern, reason, pts in FINANCIAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                snippet = match.group(0)
                detected_indicators.append({
                    "category": "Financial / Scam Lure",
                    "term": snippet,
                    "explanation": reason,
                    "points": pts
                })
                reasons.append(f"Financial lure: '{snippet}' ({reason})")
                total_points += pts

        return {
            "indicators": detected_indicators,
            "reasons": reasons,
            "points": min(total_points, 65)
        }

    def predict_ml(self, text: str) -> Dict[str, Any]:
        """
        Executes TF-IDF and Logistic Regression prediction if model is loaded.
        """
        if not self.is_model_loaded or self.model is None or self.vectorizer is None:
            return {
                "available": False,
                "label": "unclassified",
                "probability": 0.5,
                "confidence_percent": 0.0,
                "note": "Machine learning model not trained or loaded yet."
            }

        try:
            vec = self.vectorizer.transform([text])
            pred_label = self.model.predict(vec)[0]
            probs = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            
            # Probability that the message is phishing
            if "phishing" in classes:
                phish_idx = classes.index("phishing")
                phishing_prob = float(probs[phish_idx])
            else:
                phishing_prob = 1.0 if pred_label == "phishing" else 0.0

            return {
                "available": True,
                "label": str(pred_label),
                "phishing_probability": phishing_prob,
                "confidence_percent": round(float(np.max(probs)) * 100, 1),
                "note": f"ML model predicted '{pred_label}' with {round(float(np.max(probs)) * 100, 1)}% confidence."
            }
        except Exception as e:
            return {
                "available": False,
                "label": "error",
                "phishing_probability": 0.5,
                "confidence_percent": 0.0,
                "note": f"Inference error: {e}"
            }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Conducts holistic risk analysis combining ML inference, rule-based indicators,
        formatting checks, and offline URL analysis.
        
        Args:
            text (str): Input text message or email content.
            
        Returns:
            Dict[str, Any]: Full structured diagnostic report.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "empty_input": True,
                "risk_score": 0,
                "risk_category": "LOW RISK",
                "predicted_category": "None",
                "reasons": ["No message text was provided for analysis."],
                "indicators": [],
                "url_analysis": {
                    "url_count": 0,
                    "urls": [],
                    "url_findings": [],
                    "disclaimer": "No URLs detected."
                },
                "recommendation": "Paste or enter a message to begin risk scanning.",
                "ml_result": {
                    "available": False,
                    "label": "None",
                    "note": "No input provided."
                }
            }

        cleaned_text = text.strip()

        # 1. URL Analysis (safe, offline)
        url_report = analyze_urls(cleaned_text)

        # 2. Keyword & Intent Indicators
        keyword_report = self.detect_keyword_indicators(cleaned_text)

        # 3. Formatting & Caps Anomalies
        formatting_report = self.detect_formatting_anomalies(cleaned_text)

        # 4. Machine Learning Inference
        ml_report = self.predict_ml(cleaned_text)

        # 5. Hybrid Heuristic Risk Calculation
        # Compute base heuristic score (0 - 100)
        rule_points = keyword_report["points"] + formatting_report["points"] + url_report["total_url_heuristic_points"]

        if ml_report["available"]:
            ml_phish_prob = ml_report.get("phishing_probability", 0.5)
            # Combine: 45% ML probability (scaled 0-100) + 55% transparent rule-based points
            # With safeguards: strong rule points will elevate risk regardless of ML output
            combined_score = (ml_phish_prob * 45) + (rule_points * 0.75)
        else:
            # When ML is not available, scale rule points
            combined_score = rule_points * 1.1

        # High-risk trigger conditions: Brand imitation or IP host with credential request
        if url_report.get("has_brand_imitation") or url_report.get("has_ip_urls"):
            if any(ind["category"] == "Credential Harvesting" for ind in keyword_report["indicators"]):
                combined_score = max(combined_score, 85)

        # Ensure bounded between 0 and 100
        risk_score = int(min(max(round(combined_score), 0), 100))

        # 6. Risk Level Categorization
        if risk_score >= 70:
            risk_category = "HIGH RISK"
        elif risk_score >= 35:
            risk_category = "MEDIUM RISK"
        else:
            risk_category = "LOW RISK"

        # 7. Synthesize Explanations & Reasons
        all_reasons = []
        if ml_report["available"]:
            all_reasons.append(
                f"Machine Learning text model classified this message as '{ml_report['label']}' "
                f"({ml_report['confidence_percent']}% confidence)."
            )
        else:
            all_reasons.append("ML model is not loaded; evaluation is based entirely on rule-based cybersecurity indicators.")

        all_reasons.extend(keyword_report["reasons"])
        all_reasons.extend(formatting_report["flags"])
        all_reasons.extend(url_report["url_findings"])

        if not all_reasons or len(all_reasons) <= 1:
            all_reasons.append("No high-confidence suspicious indicators or anomalous patterns were detected.")

        # 8. Actionable Safety Recommendations
        if risk_category == "HIGH RISK":
            recommendation = (
                "Do NOT click any links, open attachments, or disclose passwords/personal details. "
                "Verify the sender directly through a known official phone number or trusted website."
            )
        elif risk_category == "MEDIUM RISK":
            recommendation = (
                "Exercise caution. This message exhibits some urgency or link characteristics common in scams. "
                "Confirm the request independently before taking action."
            )
        else:
            recommendation = (
                "Message appears standard with low risk indicators. However, always exercise standard vigilance "
                "and never share sensitive passcodes through unsolicited communications."
            )

        # Predicted category label for reporting
        if ml_report["available"]:
            pred_cat = ml_report["label"].capitalize()
        else:
            pred_cat = "Phishing (Heuristic)" if risk_category == "HIGH RISK" else (
                "Suspicious (Heuristic)" if risk_category == "MEDIUM RISK" else "Legitimate (Heuristic)"
            )

        return {
            "empty_input": False,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "predicted_category": pred_cat,
            "reasons": all_reasons,
            "indicators": keyword_report["indicators"],
            "formatting_flags": formatting_report["flags"],
            "url_analysis": url_report,
            "recommendation": recommendation,
            "ml_result": ml_report,
            "heuristic_breakdown": {
                "keyword_points": keyword_report["points"],
                "formatting_points": formatting_report["points"],
                "url_points": url_report["total_url_heuristic_points"],
                "combined_raw_score": round(combined_score, 1)
            }
        }
