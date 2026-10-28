"""
Unit and functional tests for url_analyzer.py
Tests safe extraction, URL shorteners, IP host detection, brand imitation,
formatting anomalies, and offline inspection constraints.
"""

import pytest
from url_analyzer import extract_urls, analyze_single_url, analyze_urls


class TestUrlAnalyzer:
    """Tests for safe offline URL analysis."""

    def test_extract_urls_multiple_and_schemeless(self):
        text = "Visit https://chase.com or http://legit.org and also check bit.ly/promo123 for details."
        urls = extract_urls(text)
        assert len(urls) >= 3
        assert "https://chase.com" in urls
        assert "http://legit.org" in urls
        assert any("bit.ly/promo123" in u for u in urls)

    def test_extract_urls_empty_and_invalid(self):
        assert extract_urls("") == []
        assert extract_urls("Hello world, no link here.") == []
        assert extract_urls(None) == []  # type: ignore

    def test_url_shortener_detection(self):
        res = analyze_single_url("https://tinyurl.com/urgent-verify")
        assert res["is_shortener"] is True
        assert any("shortener" in f.lower() for f in res["findings"])

    def test_ip_address_host_detection(self):
        res = analyze_single_url("http://192.168.1.55/login/security")
        assert res["is_ip"] is True
        assert any("raw ip" in f.lower() for f in res["findings"])

    def test_brand_imitation_detection(self):
        # Fake paypal domain
        res1 = analyze_single_url("http://paypal-security-update.click/login")
        assert any("brand imitation" in f.lower() and "paypal" in f.lower() for f in res1["findings"])

        # Leetspeak paypal domain: paypa1
        res2 = analyze_single_url("http://paypa1-support.com")
        assert any("brand imitation" in f.lower() and "paypal" in f.lower() for f in res2["findings"])

        # Genuine paypal domain should not trigger imitation flag
        res3 = analyze_single_url("https://www.paypal.com/signin")
        assert not any("brand imitation" in f.lower() for f in res3["findings"])

    def test_suspicious_tld_and_hyphens(self):
        res = analyze_single_url("http://account-verification-alert.xyz")
        assert any("high-risk top-level domain" in f.lower() for f in res["findings"])
        assert any("multiple hyphens" in f.lower() for f in res["findings"])

    def test_analyze_urls_aggregated_disclaimer(self):
        report = analyze_urls("Please check this update: http://192.168.0.1/verify")
        assert report["url_count"] == 1
        assert report["has_ip_urls"] is True
        assert "Static URL inspection" in report["disclaimer"]
        assert "offline" in report["disclaimer"].lower()

    def test_acceptance_test_at03_suspicious_url_findings(self):
        """
        AT-03: A message containing suspicious URLs should display URL findings
        and explain the relevant limitations.
        """
        msg = "URGENT: Click here to restore account: http://192.168.1.10/login and bit.ly/urgent-fix"
        report = analyze_urls(msg)

        assert report["url_count"] >= 2
        assert report["has_shorteners"] is True
        assert report["has_ip_urls"] is True
        assert len(report["url_findings"]) > 0
        # Check that disclaimer explaining static offline limitations is present
        assert "static" in report["disclaimer"].lower()
        assert "cannot" in report["disclaimer"].lower()
