"""
build_kb.py - Dubai Heritage Knowledge Base Pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from rag import chunk_text, embed_and_store

PROJECT_DIR = Path(__file__).parent
OUTPUT_FILE = PROJECT_DIR / "dubai_buildings_full.json"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_BASE = "https://en.wikipedia.org/wiki/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

HEADERS = {"User-Agent": "DubaiHistoricBuildingsStudentProject/1.0"}

SITES = [
    "Al Fahidi Fort",
    "Al Fahidi Historical Neighbourhood",
    "Sheikh Saeed Al Maktoum House",
    "Al Ahmadiya School",
    "Heritage House",
    "Al Shindagha Heritage District",
    "Dubai Creek",
    "Jumeirah Mosque",
    "Bastakiya Mosque"
]

# Fixed alternative name mapping to avoid pulling the wrong Wikipedia article
ALTERNATIVE_NAMES = {
    "al fahidi historical neighbourhood": "al bastakiya",
    "al fahidi fort": "dubai museum",
    "heritage house": "heritage house (dubai)",
    "al shindagha heritage district": "al shindagha",
    "bastakiya mosque": "al bastakiya mosque"
}

SITE_COORDINATES = {
    "Al Fahidi Fort": {"lat": 25.2635, "lon": 55.2972},
    "Al Fahidi Historical Neighbourhood": {"lat": 25.2631, "lon": 55.3003},
    "Sheikh Saeed Al Maktoum House": {"lat": 25.2694, "lon": 55.2887},
    "Al Ahmadiya School": {"lat": 25.2691, "lon": 55.2992},
    "Heritage House": {"lat": 25.2687, "lon": 55.2995},
    "Al Shindagha Heritage District": {"lat": 25.2717, "lon": 55.2870},
    "Dubai Creek": {"lat": 25.2630, "lon": 55.2980},
    "Jumeirah Mosque": {"lat": 25.2349, "lon": 55.2654},
    "Bastakiya Mosque": {"lat": 25.2629, "lon": 55.3010}
}


def wikipedia_api_get(session: requests.Session, params: dict[str, Any]) -> requests.Response:
    for attempt in range(3):
        time.sleep(0.5)
        response = session.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=20)
        if response.status_code != 429 or attempt == 2:
            response.raise_for_status()
            return response
        time.sleep(float(response.headers.get("Retry-After", "3")))
    raise RuntimeError("Wikipedia API rate limit exceeded.")


def get_full_wikipedia_page(title: str, session: requests.Session) -> dict[str, str] | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    response = wikipedia_api_get(session, params)
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})

    if "missing" in page or not page.get("extract"):
        return None

    return {
        "title": page["title"],
        "full_text": page.get("extract", "").strip()
    }


def find_wikipedia_data(site_name: str, session: requests.Session) -> dict[str, Any]:
    target_title = ALTERNATIVE_NAMES.get(site_name.lower(), site_name)
    page_data = get_full_wikipedia_page(target_title, session)

    if not page_data:
        search_res = wikipedia_api_get(
            session,
            {
                "action": "query",
                "list": "search",
                "srsearch": f"{site_name} Dubai",
                "srnamespace": 0,
                "srlimit": 1,
                "format": "json"
            }
        ).json()
        matches = search_res.get("query", {}).get("search", [])
        if matches:
            page_data = get_full_wikipedia_page(matches[0]["title"], session)

    if page_data:
        text_lines = page_data["full_text"].split("\n")
        summary = text_lines[0] if text_lines else page_data["full_text"][:300]
        return {
            "status": "found",
            "wikipedia_title": page_data["title"],
            "wikipedia_url": WIKIPEDIA_PAGE_BASE + page_data["title"].replace(" ", "_"),
            "summary": summary,
            "full_text": page_data["full_text"]
        }

    return {
        "status": "not_found",
        "wikipedia_title": None,
        "wikipedia_url": None,
        "summary": "No Wikipedia text found.",
        "full_text": ""
    }


def fetch_commons_images(site_name: str, session: requests.Session, max_results: int = 3) -> list[dict[str, str]]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{site_name} Dubai",
        "gsrnamespace": 6,
        "gsrlimit": max_results,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1200,
        "format": "json",
    }
    time.sleep(0.5)
    try:
        res = session.get(COMMONS_API, params=params, headers=HEADERS, timeout=20)
        res.raise_for_status()
        pages = res.json().get("query", {}).get("pages", {})
        images = []
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            metadata = info.get("extmetadata", {})
            images.append({
                "title": page.get("title", "Image").replace("File:", ""),
                "image_url": info.get("thumburl", info.get("url", "")),
                "file_page_url": info.get("descriptionurl", ""),
                "license": metadata.get("LicenseShortName", {}).get("value", "CC BY-SA"),
                "artist": metadata.get("Artist", {}).get("value", "Wikimedia Commons")
            })
        return images
    except Exception as e:
        print(f"  Image lookup failed for {site_name}: {e}")
        return []


def main():
    session = requests.Session()
    dataset = []

    print("--- Harvesting Heritage Sites & Coordinates ---")
    for site in SITES:
        print(f"Fetching: {site}...")
        wiki_data = find_wikipedia_data(site, session)
        images = fetch_commons_images(site, session, max_results=3)
        coords = SITE_COORDINATES.get(site, {"lat": 25.2631, "lon": 55.3003})

        record = {
            "site_name": site,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "wikipedia": wiki_data,
            "images": images
        }
        dataset.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved dataset JSON to {OUTPUT_FILE}")

    print("\n--- Chunking & Ingesting into Vector Store ---")
    total_chunks = 0
    for record in dataset:
        full_text = record["wikipedia"]["full_text"]
        if not full_text:
            continue
        chunks = chunk_text(full_text, size=500, overlap=50)
        stored_count = embed_and_store(chunks, filename=record["site_name"])
        total_chunks += stored_count
        print(f"  -> Ingested {stored_count} chunks for {record['site_name']}")

    print(f"\nCompleted! Stored {total_chunks} total chunks.")


if __name__ == "__main__":
    main()