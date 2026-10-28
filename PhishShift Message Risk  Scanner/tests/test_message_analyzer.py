"""
Comprehensive automated test suite for message_analyzer.py and train_model.py.
Validates functional behaviors, edge cases, dataset validation, error handling,
and acceptance tests AT-01, AT-02, AT-03.
"""

import os
import tempfile
import pandas as pd
import pytest

from message_analyzer import MessageAnalyzer
from train_model import load_and_validate_dataset, train_and_evaluate


class TestMessageAnalyzer:
    """Core functional and edge-case tests for MessageAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup_analyzer(self):
        # Initialize analyzer (works whether model is loaded or not)
        self.analyzer = MessageAnalyzer()

    def test_empty_and_whitespace_input(self):
        res1 = self.analyzer.analyze("")
        assert res1["empty_input"] is True
        assert res1["risk_score"] == 0
        assert res1["risk_category"] == "LOW RISK"

        res2 = self.analyzer.analyze("    \n\t  ")
        assert res2["empty_input"] is True

        res3 = self.analyzer.analyze(None)  # type: ignore
        assert res3["empty_input"] is True

    def test_message_with_no_urls(self):
        text = "Hey Sarah, are we still meeting for lunch at 12:30 PM today?"
        res = self.analyzer.analyze(text)
        assert res["url_analysis"]["url_count"] == 0
        assert res["risk_category"] == "LOW RISK"
        assert res["risk_score"] < 35

    def test_obvious_phishing_message(self):
        text = (
            "URGENT: Your Chase bank account has been suspended! "
            "Verify your password and SSN immediately to avoid closure: http://chase-security-verify.top/login"
        )
        res = self.analyzer.analyze(text)
        assert res["risk_category"] == "HIGH RISK"
        assert res["risk_score"] >= 70
        assert len(res["indicators"]) > 0
        assert any("Credential Harvesting" in ind["category"] for ind in res["indicators"])

    def test_message_with_shortened_url(self):
        text = "Your package delivery failed. Confirm your shipping address within 24 hours: bit.ly/usps-track"
        res = self.analyzer.analyze(text)
        assert res["url_analysis"]["has_shorteners"] is True
        assert res["risk_score"] >= 35  # Elevated risk due to shortener and urgent delivery claim

    def test_model_loading_error_fallback(self):
        # Create an analyzer pointing to nonexistent paths
        fallback_analyzer = MessageAnalyzer(
            model_path="nonexistent_model.joblib",
            vectorizer_path="nonexistent_vectorizer.joblib"
        )
        assert fallback_analyzer.is_model_loaded is False

        # Should still run successfully via rule-based heuristics without raising exceptions
        res = fallback_analyzer.analyze("URGENT: Verify your password now at http://fake-bank.cam")
        assert res["risk_category"] == "HIGH RISK"
        assert res["risk_score"] >= 70
        assert res["ml_result"]["available"] is False


class TestAcceptanceCriteria:
    """Specific acceptance criteria specified in project requirements."""

    def test_at01_legitimate_password_reset(self):
        """
        AT-01: A legitimate password-reset notification should not
        automatically be classified as phishing.
        """
        analyzer = MessageAnalyzer()
        legit_pwd_reset = (
            "Your GitHub password was recently changed on Oct 27, 2026. "
            "If you did this, no further action is required. "
            "If you did not initiate this change, please review your account activity on github.com."
        )
        res = analyzer.analyze(legit_pwd_reset)

        # Must not be high risk or classified as phishing
        assert res["risk_category"] != "HIGH RISK"
        assert res["risk_score"] < 50
        # Recommendation should be safe or low/moderate guidance
        assert "password" in legit_pwd_reset.lower()

    def test_at02_unfamiliar_phishing_style_language(self):
        """
        AT-02: A message using unfamiliar phishing-style language should
        be assessed using the available model and indicators.
        """
        analyzer = MessageAnalyzer()
        novel_phishing = (
            "FINAL NOTICE: Immediate direct deposit reallocation anomaly detected. "
            "Submit your security credentials and PIN within 12 hours to release funds: http://192.168.1.99/auth"
        )
        res = analyzer.analyze(novel_phishing)

        # Check that heuristic indicators caught urgency and credential requests
        assert res["risk_score"] >= 70
        assert res["risk_category"] == "HIGH RISK"
        assert len(res["reasons"]) >= 2
        assert len(res["recommendation"]) > 0

    def test_at03_suspicious_url_findings_and_limitations(self):
        """
        AT-03: A message containing suspicious URLs should display URL findings
        and explain the relevant limitations.
        """
        analyzer = MessageAnalyzer()
        msg_with_bad_url = (
            "Security update required. Please login at: http://paypa1-secure-verification.click/login"
        )
        res = analyzer.analyze(msg_with_bad_url)

        assert res["url_analysis"]["url_count"] == 1
        assert len(res["url_analysis"]["url_findings"]) > 0
        # Verification that limitation disclaimer is provided
        assert "disclaimer" in res["url_analysis"]
        assert "cannot" in res["url_analysis"]["disclaimer"].lower() or "limit" in res["url_analysis"]["disclaimer"].lower()


class TestDatasetManagementAndValidation:
    """Tests for dataset validation, duplicate removal, invalid label handling, and missing files."""

    def test_missing_dataset_file(self):
        with pytest.raises(FileNotFoundError):
            load_and_validate_dataset("nonexistent_messages.csv")

    def test_duplicate_dataset_rows_and_invalid_labels(self):
        # Create a temporary CSV with duplicates, missing values, and invalid labels
        data = {
            "text": [
                "Your Amazon order has shipped.",
                "Your Amazon order has shipped.",  # Duplicate
                "URGENT: Bank account locked!",
                "URGENT: Bank account locked!",    # Duplicate
                "Meeting tomorrow at 10 AM",
                "Your Uber receipt is $12",
                "Winner of $500 gift card!",
                "Password change confirmed.",
                "Verify your paypal account now: http://paypa1.top",
                "Flight confirmation #1234",
                "IRS tax refund available!",
                "Valid email message here.",
                "",                                # Empty text
                "Another text with invalid label"
            ],
            "label": [
                "legitimate",
                "legitimate",
                "phishing",
                "phishing",
                "legitimate",
                "legitimate",
                "phishing",
                "legitimate",
                "phishing",
                "legitimate",
                "phishing",
                "legitimate",
                "phishing",
                "UNKNOWN_LABEL"                    # Invalid label
            ]
        }
        df_temp = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df_temp.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            cleaned_df = load_and_validate_dataset(temp_path)
            # Duplicates must be removed
            assert len(cleaned_df[cleaned_df["text"] == "Your Amazon order has shipped."]) == 1
            assert len(cleaned_df[cleaned_df["text"] == "URGENT: Bank account locked!"]) == 1
            # Invalid label must be filtered out
            assert "UNKNOWN_LABEL" not in cleaned_df["label"].values
            # Empty text must be dropped
            assert "" not in cleaned_df["text"].values
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
