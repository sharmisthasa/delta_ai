"""Refresh the image galleries in dubai_buildings_full.json.

Fills in sites that had no or unsuitable imagery on Wikimedia Commons and
swaps out a few wrong results, so every heritage site has a coherent set of
square-friendly photos. Run from the wiki page folder:

    .venv/bin/python refresh_images.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "dubai_buildings_full.json"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "SikkaDubaiHeritageArchive/1.0 (student project)"}

# Exact Wikimedia Commons file titles to use per site. Sites not listed keep
# the images already in the dataset.
CURATED_IMAGES = {
    "Al Fahidi Fort": [
        "Al Fahidi-Fort.jpg",
        "Al Fahidi Fort in Dubai by Vincent Eisfeld.jpg",
        "UAE Dubai Al Fahidi Fort img1 asv2018-01.jpg",
        "Courtyard of Al Fahidi Fort (8667298481).jpg",
    ],
    "Al Maktoum Residence": [
        "Museum house Al Shindagha.jpg",
        "Shindagha stretchoftraditionalhomes.jpg",
        "House of H.H sheikh Saeed Al Maktoum.jpg",
        "UAE Dubai Shindagha village img1 asv2018-01.jpg",
    ],
    "Al Ahmadiya School": [
        "Al Ahmadiya School.JPG",
        "Al Ahmadiya School2.JPG",
        "Al Ahmadiya 1.jpg",
        "Al Ahmadiya 2.jpg",
    ],
    "Majlis Ghorfat Umm Al Sheif": [
        "Majlis Ghorfat 1 Umm Al Sheif.jpg",
        "Majlis Ghorfat 2 Umm Al Sheif.jpg",
        "Majlis Ghorfat 3 Umm Al Sheif.jpg",
        "Majlis Ghorfat 4 Umm Al Sheif.jpg",
    ],
    # The tower itself has no freely-licensed photograph on Commons, so give
    # the record its Deira / creek setting instead of leaving it image-less.
    "Burj Nahar Watchtower": [
        "Al Ras.jpg",
        "The line between old Dubai and new Dubai.jpg",
        "Abra -Dubai Creek Old boats.jpg",
    ],
}


def _commons_get(params: dict) -> dict:
    for attempt in range(4):
        time.sleep(0.8)
        response = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=20)
        if response.status_code == 429 and attempt < 3:
            time.sleep(float(response.headers.get("Retry-After", "5")))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Wikimedia Commons rate limit exceeded.")


def image_info(title: str) -> dict | None:
    """Fetch the same shape of metadata the pipeline stores for one file title."""
    data = _commons_get(
        {
            "action": "query",
            "titles": f"File:{title}",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1200,
            "format": "json",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "imageinfo" not in page:
            continue
        info = page["imageinfo"][0]
        metadata = info.get("extmetadata", {})
        return {
            "title": title,
            "image_url": info.get("thumburl") or info.get("url", ""),
            "file_page_url": info.get("descriptionurl", ""),
            "license": metadata.get("LicenseShortName", {}).get("value", "CC BY-SA"),
            "artist": metadata.get("Artist", {}).get("value", "Wikimedia Commons"),
        }
    return None


def main() -> None:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for record in dataset:
        site_name = record["site_name"]
        wanted = CURATED_IMAGES.get(site_name)
        if wanted is None:
            continue

        fresh = []
        for title in wanted:
            info = image_info(title)
            if info and info["image_url"]:
                fresh.append(info)
                print(f"  + {site_name}: {title}")
            else:
                print(f"  ! {site_name}: could not resolve {title}")

        if site_name in CURATED_IMAGES and fresh:
            record["images"] = fresh
        else:
            print(f"  ! {site_name}: no images resolved, leaving existing gallery.")

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("\nDone. Run the app to see the updated galleries.")


if __name__ == "__main__":
    main()