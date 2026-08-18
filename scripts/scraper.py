"""
PakWheels Used Car Scraper
===========================
Two-stage scraper:
  Stage 1: Scrape search-result pages. Each listing card embeds a
           schema.org JSON-LD block with clean structured data
           (make, model, year, price, mileage, fuel, transmission, engine cc).
  Stage 2: Visit each listing's detail page to pull the fields that are
           NOT in the JSON-LD: assembly status (Local/Imported) and
           registered-in city/province. These are the Pakistan-specific
           features that make this project's dataset unique.

Usage:
    python scraper.py --pages 50 --output cars_raw.csv
    python scraper.py --pages 50 --city lahore --output lahore_cars.csv
    python scraper.py --resume cars_raw.csv --pages 50   # skip already-scraped IDs

Be a good citizen: this script rate-limits itself (default 1.5s between
requests) and uses a single persistent session. Don't lower the delay
aggressively — PakWheels will start blocking / serving CAPTCHAs.
"""

import argparse
import csv
import json
import os
import random
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.pakwheels.com"
SEARCH_URL_TMPL = BASE_URL + "/used-cars/search/-/{filters}?page={page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FIELDNAMES = [
    "listing_id", "url", "title", "make", "model", "year", "price_pkr",
    "mileage_km", "fuel_type", "transmission", "engine_cc", "city",
    "assembly", "registered_in", "seller_type", "scraped_at",
]


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def polite_sleep(base=1.5, jitter=0.8):
    """Sleep base seconds plus random jitter, so requests aren't robotically evenly spaced."""
    time.sleep(base + random.uniform(0, jitter))


def parse_search_page(html):
    """
    Extract listing cards from a search-results page using the embedded
    JSON-LD schema.org blocks. Returns a list of dicts (partial rows —
    missing assembly/registered_in, filled in stage 2).
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # Each <li class="classified-listing ..."> contains one <script type="application/ld+json">
    listings = soup.select("li.classified-listing")
    for li in listings:
        script = li.find("script", {"type": "application/ld+json"})
        if not script or not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        offer = data.get("offers", {})
        url = offer.get("url", "")
        listing_id = None
        m = re.search(r"-(\d+)$", url)
        if m:
            listing_id = m.group(1)

        engine = data.get("vehicleEngine", {}) or {}
        engine_cc = engine.get("engineDisplacement", "")
        engine_cc = re.sub(r"[^\d]", "", str(engine_cc)) or None

        mileage_raw = data.get("mileageFromOdometer", "")
        mileage_km = re.sub(r"[^\d]", "", str(mileage_raw)) or None

        # City: not in JSON-LD directly, but is in the "description" field
        # ("Honda Civic 2015 for sale in Lahore") — extract it.
        city = None
        desc = data.get("description", "")
        m_city = re.search(r"for sale in ([A-Za-z ]+)$", desc)
        if m_city:
            city = m_city.group(1).strip()

        rows.append({
            "listing_id": listing_id,
            "url": url,
            "title": data.get("name", ""),
            "make": (data.get("brand") or {}).get("name", ""),
            "model": "",  # filled from title parsing if needed
            "year": data.get("modelDate", ""),
            "price_pkr": offer.get("price", ""),
            "mileage_km": mileage_km,
            "fuel_type": data.get("fuelType", ""),
            "transmission": data.get("vehicleTransmission", ""),
            "engine_cc": engine_cc,
            "city": city,
            "assembly": None,
            "registered_in": None,
            "seller_type": None,
            "scraped_at": int(time.time()),
        })

    # Figure out if there's a next page: PakWheels shows "X - Y of N Results"
    total_results = None
    info = soup.select_one(".search-pagi-info")
    if info:
        m = re.search(r"of\s*<b>([\d,]+)</b>", str(info))
        if m:
            total_results = int(m.group(1).replace(",", ""))

    return rows, total_results


def parse_detail_page(html):
    """
    Extract assembly status, registered-in city, and seller type from a
    single listing's detail page.

    PakWheels renders the "General Info" block as flat, alternating <li>
    pairs inside a <ul>:
        <li class="ad-data">Assembly</li>
        <li><a ...>Imported</a></li>     (or plain text, no <a>)
        <li class="ad-data">Engine Capacity</li>
        <li>1500 cc</li>
        ...
    i.e. every <li class="ad-data"> is a label, and the very next <li>
    sibling (which does NOT have class="ad-data") is its value. We walk
    all <li> tags in document order and pair them up generically -- this
    way we capture every label PakWheels exposes (Assembly, Registered
    City/Registered In, Color, Body Type, etc.) without hardcoding each
    one, and it keeps working even if the exact "registered" label
    wording changes slightly.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {"assembly": None, "registered_in": None, "seller_type": None}

    fields = {}
    all_li = soup.find_all("li")
    for i, li in enumerate(all_li):
        classes = li.get("class") or []
        if "ad-data" in classes:
            label = li.get_text(strip=True).rstrip(":").lower()
            if i + 1 < len(all_li):
                value_li = all_li[i + 1]
                value = value_li.get_text(strip=True)
                if value:
                    fields[label] = value

    for label, value in fields.items():
        if "assembly" in label:
            result["assembly"] = value
        elif "registered" in label:
            result["registered_in"] = value

    # Seller type: dealer listings link to a /dealers/ or /showrooms/ page
    # somewhere on the detail page; individual sellers don't.
    if soup.select_one("a[href*='/dealers/'], a[href*='/showrooms/'], .dealer-badge"):
        result["seller_type"] = "Dealer"
    else:
        result["seller_type"] = "Individual"

    return result


def load_scraped_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("listing_id"):
                ids.add(row["listing_id"])
    return ids


def scrape(pages, output, filters="", delay=1.5, fetch_details=True, resume=False):
    session = get_session()
    seen_ids = load_scraped_ids(output) if resume else set()
    write_header = not (resume and os.path.exists(output))

    mode = "a" if (resume and os.path.exists(output)) else "w"
    with open(output, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for page in range(1, pages + 1):
            url = SEARCH_URL_TMPL.format(filters=filters, page=page)
            print(f"[page {page}] GET {url}")
            try:
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  ! request failed: {e} -- skipping page")
                polite_sleep(delay)
                continue

            rows, total_results = parse_search_page(resp.text)
            if not rows:
                print("  ! no listings found on this page -- stopping (likely hit end or a block page)")
                break

            new_count = 0
            for row in rows:
                if row["listing_id"] in seen_ids:
                    continue
                seen_ids.add(row["listing_id"])

                if fetch_details and row["url"]:
                    polite_sleep(delay)
                    try:
                        d_resp = session.get(urljoin(BASE_URL, row["url"]), timeout=20)
                        d_resp.raise_for_status()
                        details = parse_detail_page(d_resp.text)
                        row.update(details)
                    except requests.RequestException as e:
                        print(f"  ! detail fetch failed for {row['url']}: {e}")

                writer.writerow(row)
                new_count += 1

            f.flush()
            print(f"  -> {new_count} new rows written (total results on site: {total_results})")
            polite_sleep(delay)

    print(f"Done. Output written to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape PakWheels used car listings")
    ap.add_argument("--pages", type=int, default=10, help="number of search-result pages to scrape (25 listings/page)")
    ap.add_argument("--output", type=str, default="cars_raw.csv")
    ap.add_argument(
        "--filters", type=str, default="",
        help="PakWheels filter path segment, e.g. 'ctr_japanese/' -- leave blank for all cars",
    )
    ap.add_argument("--delay", type=float, default=1.5, help="base delay between requests, in seconds")
    ap.add_argument(
        "--no-details", action="store_true",
        help="skip stage 2 (detail-page) scraping -- faster, but no assembly/registered_in",
    )
    ap.add_argument("--resume", action="store_true", help="append to --output, skipping listing_ids already present")
    args = ap.parse_args()

    scrape(
        pages=args.pages,
        output=args.output,
        filters=args.filters,
        delay=args.delay,
        fetch_details=not args.no_details,
        resume=args.resume,
    )
