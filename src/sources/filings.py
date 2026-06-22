
import requests
from src.cache import cached

SEC_HEADERS = {"User-Agent": "AlphaLens rayyan.suhail2001@gmail.com"}
USEFUL_FORMS = ["10-K", "10-Q", "8-K"]

@cached(ttl=86400)
def get_cik(ticker):
    """Look up a ticker's zero-padded 10-digit CIK from the SEC's mapping."""
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=10,
    )
    if response.status_code != 200:
        raise ValueError(f"SEC ticker lookup failed: status {response.status_code}")

    mapping = response.json()

    for entry in mapping.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"No CIK found for ticker '{ticker}'.")


@cached(ttl=86400)
def get_filings(ticker, limit=5):
    """Fetch a company's recent important SEC filings (10-K, 10-Q, 8-K)."""
    cik = get_cik(ticker)

    response = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=10,
    )
    if response.status_code != 200:
        raise ValueError(f"SEC filings lookup failed: status {response.status_code}")

    data = response.json()
    recent = data["filings"]["recent"]

    forms = recent["form"]
    dates = recent["filingDate"]
    accessions = recent["accessionNumber"]
    primary_docs = recent["primaryDocument"]

    filings = []
    for i in range(len(forms)):
        if forms[i] not in USEFUL_FORMS:
            continue

        accession_clean = accessions[i].replace("-", "")
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_clean}/{primary_docs[i]}"
        )

        filings.append({
            "form": forms[i],
            "date": dates[i],
            "url": doc_url,
        })

        if len(filings) >= limit:
            break

    return filings