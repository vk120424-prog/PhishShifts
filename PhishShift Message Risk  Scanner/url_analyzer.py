"""
PhishShift - URL Analyzer Module
Provides safe, offline inspection of URLs extracted from text messages.
Does not make network requests or attempt to resolve domains.
"""

import re
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional

# Known URL shortener services commonly leveraged in phishing/smishing
KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "buff.ly", "ow.ly",
    "rb.gy", "goo.gl", "cutt.ly", "rebrand.ly", "shorturl.at",
    "tiny.cc", "bc.vc", "shorte.st", "adf.ly", "v.gd"
}

# Suspicious or high-risk TLDs frequently abused in malicious campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "cam", "work",
    "click", "party", "racing", "bid", "loan", "stream", "live", "country",
    "rest", "gdn", "date", "faith", "review", "zip", "mov"
}

# High-profile brands frequently targeted by phishing campaigns
TARGETED_BRANDS = {
    "paypal": ["paypal.com"],
    "google": ["google.com", "accounts.google.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "bankofamerica": ["bankofamerica.com"],
    "citibank": ["citi.com", "citibank.com"],
    "facebook": ["facebook.com", "meta.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com"],
    "twitter": ["twitter.com", "x.com"],
    "dropbox": ["dropbox.com"],
    "fedex": ["fedex.com"],
    "dhl": ["dhl.com"],
    "usps": ["usps.com"]
}

# Common leet-speak replacements used in deceptive domains
LEET_MAPPINGS = {
    "0": "o",
    "1": "l", # or i
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a"
}

# Comprehensive URL extraction pattern (captures http/https/www with domain or IP)
URL_REGEX = re.compile(
    r'(?:(?:https?://)|(?:www\.))'  # http://, https://, or www.
    r'(?:(?:\d{1,3}\.){3}\d{1,3}|(?:[a-zA-Z0-9_\-]+\.)+[a-zA-Z]{2,})'  # IP address or domain with extension
    r'(?::\d{1,5})?'  # optional port
    r'(?:/[^\s<>"\'{}|\\^`]*)*',  # path
    re.IGNORECASE
)

# Pattern for IP addresses as hostnames (e.g., http://192.168.1.1/login)
IP_HOST_REGEX = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)


def extract_urls(text: str) -> List[str]:
    """
    Safely extract candidate URLs from raw message text without executing or opening them.
    
    Args:
        text (str): Input email or SMS text.
        
    Returns:
        List[str]: List of extracted URL strings.
    """
    if not text or not isinstance(text, str):
        return []
    
    matches = URL_REGEX.findall(text)
    # Also look for standalone shorteners like bit.ly/abcde without http
    shortener_raw_regex = re.compile(
        r'\b(?:' + '|'.join(re.escape(s) for s in KNOWN_SHORTENERS) + r')/[a-zA-Z0-9_\-]+',
        re.IGNORECASE
    )
    additional_matches = shortener_raw_regex.findall(text)
    
    all_urls = list(dict.fromkeys(matches + additional_matches))
    return all_urls


def _normalize_leet(domain: str) -> str:
    """Normalize leet-speak characters in a domain for brand imitation heuristics."""
    normalized = domain.lower()
    for char, rep in LEET_MAPPINGS.items():
        normalized = normalized.replace(char, rep)
    return normalized


def analyze_single_url(raw_url: str) -> Dict[str, Any]:
    """
    Perform deep static, offline inspection on a single URL.
    
    Args:
        raw_url (str): The raw URL string.
        
    Returns:
        Dict[str, Any]: Analysis details including flags, heuristic score, and explanation.
    """
    findings: List[str] = []
    heuristic_points = 0
    url_to_parse = raw_url
    if not url_to_parse.startswith(("http://", "https://")):
        url_to_parse = "http://" + url_to_parse

    parsed = urlparse(url_to_parse)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Strip port if present
    host = netloc.split(":")[0] if ":" in netloc else netloc

    # 1. Check for IP address as host
    is_ip = bool(IP_HOST_REGEX.match(host))
    if is_ip:
        findings.append("Uses raw IP address instead of registered domain name")
        heuristic_points += 40

    # 2. Check for URL shortener
    is_shortener = host in KNOWN_SHORTENERS or any(host.endswith("." + s) for s in KNOWN_SHORTENERS)
    if is_shortener:
        findings.append(f"Employs known URL shortener service ({host}) which conceals destination")
        heuristic_points += 25

    # 3. Check for credentials or '@' symbol in URL authority
    if "@" in raw_url:
        findings.append("Contains '@' symbol in authority section, commonly used to trick visual inspection")
        heuristic_points += 35

    # 4. Check for high-risk TLD
    tld = host.split(".")[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        findings.append(f"Uses high-risk top-level domain (.{tld}) frequently abused in phishing")
        heuristic_points += 20

    # 5. Excessive subdomains (more than 3 dot-separated parts for regular domains)
    domain_parts = host.split(".")
    if len(domain_parts) > 4 and not is_ip:
        findings.append(f"Excessive subdomains ({len(domain_parts) - 2} levels) detected")
        heuristic_points += 15

    # 6. Excessive hyphens in hostname
    if host.count("-") >= 2:
        findings.append(f"Multiple hyphens ({host.count('-')}) in domain name, typical of scam spoofing")
        heuristic_points += 15

    # 7. Brand imitation / typosquatting heuristic
    normalized_host = _normalize_leet(host)
    for brand, legit_domains in TARGETED_BRANDS.items():
        is_legit = any(host == legit or host.endswith("." + legit) for legit in legit_domains)
        if not is_legit:
            # Check if brand appears in host or normalized host
            if brand in host or brand in normalized_host:
                findings.append(
                    f"Possible brand imitation detected: references '{brand}' but domain is not an official {brand} domain ({host})"
                )
                heuristic_points += 45
                break

    # 8. Suspicious path keywords (login, verify, secure, update, banking)
    suspicious_path_terms = ["login", "signin", "verify", "secure", "update", "banking", "wallet", "account"]
    found_path_terms = [term for term in suspicious_path_terms if term in path]
    if found_path_terms and not is_shortener:
        findings.append(f"Suspicious path keywords detected: {', '.join(found_path_terms)}")
        heuristic_points += 10

    return {
        "url": raw_url,
        "host": host,
        "is_ip": is_ip,
        "is_shortener": is_shortener,
        "heuristic_points": min(heuristic_points, 85),
        "findings": findings
    }


def analyze_urls(text: str) -> Dict[str, Any]:
    """
    Extracts all URLs and produces an aggregated safe offline analysis report.
    
    Args:
        text (str): The submitted message.
        
    Returns:
        Dict[str, Any]: Aggregated URL analysis findings.
    """
    urls = extract_urls(text)
    
    if not urls:
        return {
            "url_count": 0,
            "urls": [],
            "has_shorteners": False,
            "has_ip_urls": False,
            "has_brand_imitation": False,
            "total_url_heuristic_points": 0,
            "url_findings": [],
            "disclaimer": "No URLs were detected in the message. Offline static checks only."
        }

    detailed_results = []
    combined_findings = []
    has_shorteners = False
    has_ip_urls = False
    has_brand_imitation = False
    total_points = 0

    for u in urls:
        res = analyze_single_url(u)
        detailed_results.append(res)
        combined_findings.extend(res["findings"])
        if res["is_shortener"]:
            has_shorteners = True
        if res["is_ip"]:
            has_ip_urls = True
        if any("brand imitation" in f for f in res["findings"]):
            has_brand_imitation = True
        total_points += res["heuristic_points"]

    # Cap aggregated URL points to a sensible maximum contribution (e.g. 70)
    capped_points = min(total_points, 70)

    return {
        "url_count": len(urls),
        "urls": urls,
        "details": detailed_results,
        "has_shorteners": has_shorteners,
        "has_ip_urls": has_ip_urls,
        "has_brand_imitation": has_brand_imitation,
        "total_url_heuristic_points": capped_points,
        "url_findings": list(dict.fromkeys(combined_findings)),
        "disclaimer": (
            "Static URL inspection analyses structural heuristics and domain syntax offline. "
            "It cannot inspect live server content, redirects, or verify whether a website is benign."
        )
    }
