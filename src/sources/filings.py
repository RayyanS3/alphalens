
import requests
from src.cache import cached
from bs4 import BeautifulSoup


SEC_HEADERS = {"User-Agent": "AlphaLens rayyan.suhail2001@gmail.com"}
USEFUL_FORMS = ["10-K", "10-Q", "8-K"]
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


@cached(ttl=86400)
def get_cik(ticker: str) -> str:
    response = requests.get(
        SEC_TICKERS_URL,
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
def get_filings(ticker: str, limit: int = 5) -> list[dict]:
    cik = get_cik(ticker)

    response = requests.get(
        SEC_SUBMISSIONS_URL.format(cik=cik),
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

def get_filing_text(url: str, max_chars: int = 50000) -> str:
    response = requests.get(url, headers=SEC_HEADERS, timeout=20)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch filing: status {response.status_code}")

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "head", "title", "meta"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    text = " ".join(text.split())

    return text[:max_chars]
