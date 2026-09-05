import base64
import datetime
import html
import json
import re
import warnings as _warnings
from pathlib import Path
import streamlit as st

# Interactive map support (folium + streamlit-folium)
try:
    import folium
    from streamlit_folium import st_folium

    _map_library_available = True
    _map_error = None
except ImportError as _exc:
    folium = None
    st_folium = None
    _map_library_available = False
    _map_error = str(_exc)

# ==============================================================================
# 1. INTEGRATION WITH YOUR LOCAL RAG PIPELINE (rag.py)
# ==============================================================================
try:
    from rag import answer_with_sources as _rag_answer
    from rag import get_groq_client as _get_groq_client

    _rag_import_error = None
    _vision_available = True
except ImportError as e:
    _rag_answer = None
    _get_groq_client = None
    _rag_import_error = str(e)
    _vision_available = False

# Groq vision model used for photo-based archive search (multimodal).
# Qwen 3.8 27B answers directly (no long thinking pass), keeping the free-tier
# output-token budget intact.
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_PHOTO_BYTES = 15 * 1024 * 1024

try:
    _warnings.filterwarnings("ignore", message=".*renamed.*")
    from ddgs import DDGS

    _ddgs = DDGS()
except Exception:
    _ddgs = None


def _web_search_fallback(query):
    """Search the web via DuckDuckGo when the archive can't answer."""
    if _ddgs is None:
        return None
    variants = [f"{query} Dubai heritage architecture", query]
    for attempt in range(2):
        for variant in variants:
            try:
                results = list(_ddgs.text(variant, max_results=4))
            except Exception:
                results = []
            if not results:
                continue
            snippets = []
            for r in results:
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                href = r.get("href") or ""
                if len(body) > 500:
                    body = body[:497].rstrip() + "..."
                snippets.append(f"**{title}**\n{body}\nSource: {href}")
            return "\n\n---\n\n".join(snippets)
    return None


_NOT_AVAILABLE_MARKERS = ("sorry", "couldn't find", "not available", "unable to find",
                           "i don't have", "i do not have", "no reliable")


def query_rag_safe(query: str, site_id: str = None):
    """Safely routes queries through your local RAG pipeline, with web fallback."""
    if _rag_answer is not None:
        try:
            result = _rag_answer(query, site_id=site_id)
        except TypeError:
            try:
                result = _rag_answer(query)
            except Exception as e:
                return f"Error querying local archive pipeline: {str(e)}", []
        except Exception as e:
            return f"Error querying local archive pipeline: {str(e)}", []

        # Current rag.py returns (answer, sources, status); older copies return
        # (answer, sources), so infer the status from the reply text as a fallback.
        try:
            answer, sources, status = result
        except ValueError:
            answer, sources = result
            lower = answer.lower().strip()
            status = (
                "notavailable"
                if not sources and any(marker in lower for marker in _NOT_AVAILABLE_MARKERS)
                else "archive"
            )

        # If the archive + Wikipedia couldn't answer, try the web
        needs_web = status == "notavailable"
        if not needs_web and not sources:
            lower = answer.lower().strip()
            needs_web = any(marker in lower for marker in _NOT_AVAILABLE_MARKERS)
        if needs_web:
            web_snippet = _web_search_fallback(query)
            if web_snippet:
                return (
                    f"Here's what I found online about your question:\n\n"
                    f"{web_snippet}",
                    [],
                )
        return answer, sources
    else:
        return (
            f"Archive pipeline not connected. Importing rag.py failed with: "
            f"{_rag_import_error}",
            [],
        )


# ==============================================================================
# 1b. PHOTO SEARCH (vision) — identify which heritage site a photo shows
# ==============================================================================
VISION_PROMPT = """You are a heritage photo archivist for Dubai's Athar archive.
The archive holds records for these sites:
{site_list}

Look at the photo. Decide which ONE site from the list it most likely shows.
Reply with ONLY the exact site name verbatim — no punctuation, no explanation.
If the photo does not clearly match any site in the list, reply with exactly: none"""


def identify_site_from_image(file_bytes: bytes, mime: str = "image/jpeg"):
    """Return (site, error) where site is a matched SITES record or None."""
    if not _vision_available or _get_groq_client is None:
        return (
            None,
            "Photo search is unavailable because the RAG pipeline isn't connected.",
        )
    if len(file_bytes) > MAX_PHOTO_BYTES:
        return None, "That photo is too large — please upload one under 15 MB."

    try:
        site_list = "\n".join(f"- {s['name']}: {s['blurb']}" for s in SITES)
        client = _get_groq_client()
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=60,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT.format(site_list=site_list)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,"
                                + base64.b64encode(file_bytes).decode("ascii")
                            },
                        },
                    ],
                }
            ],
        )
        reply = (completion.choices[0].message.content or "").strip().strip('".')
        reply_lower = reply.lower()
        for site in SITES:
            name_lower = site["name"].lower()
            if name_lower in reply_lower or reply_lower in name_lower:
                return site, None
        return None, None
    except Exception as e:
        return None, f"Sorry, I couldn't analyse that photo: {e}"


# ==============================================================================
# 2. LOCAL 10-BUILDING DATASET LOADER
# ==============================================================================
DATA_FILE = Path(__file__).parent / "dubai_buildings_full.json"

# Card copy is written by hand: several Wikipedia summaries in the dataset
# describe the wrong subject (the airport, a ruler's biography, a tourism
# overview), so they cannot be truncated into a trustworthy one-liner.
SITE_CARDS = {
    "Al Fahidi Fort": (
        "Al Fahidi",
        "Dubai's oldest standing building, raised in 1787 to guard the creek "
        "and now home to the Dubai Museum.",
    ),
    "Al Fahidi Historical Neighbourhood": (
        "Al Fahidi",
        "A restored quarter of coral and gypsum courtyard houses threaded with "
        "narrow lanes, settled by Persian merchants from Bastak.",
    ),
    "Sheikh Saeed Al Maktoum House": (
        "Al Shindagha",
        "The 1896 creekside home of Dubai's longest serving ruler, now a "
        "museum of photographs, documents, and coins.",
    ),
    "Al Ahmadiya School": (
        "Al Ras",
        "Dubai's first formal school, founded in 1912 and restored as a "
        "museum of the city's education history.",
    ),
    "Heritage House": (
        "Al Ras",
        "An 1890s pearl merchant's home arranged around a courtyard, open "
        "today as a museum of old family life.",
    ),
    "Al Shindagha Heritage District": (
        "Al Shindagha",
        "The restored creekside quarter of the ruling family, wrapped around "
        "wind-tower houses, the Al Shindagha Museum and Sheikh Saeed's home.",
    ),
    "Dubai Creek": (
        "Bur Dubai & Deira",
        "The saltwater inlet that split old Dubai into Deira and Bur Dubai "
        "and powered its centuries of pearling and dhow trade.",
    ),
    "Jumeirah Mosque": (
        "Jumeirah",
        "Sheikh Rashid's 1979 gift to his son: a white-stone Fatimid-style "
        "mosque and home of the 'Open Doors. Open Minds.' tours.",
    ),
    "Bastakiya Mosque": (
        "Al Fahidi",
        "A coral-and-gypsum neighbourhood mosque in the heart of the "
        "Bastakiya quarter, built with the district in the 1890s.",
    ),
    "Majlis Ghorfat Umm Al Sheif": (
        "Jumeirah",
        "Sheikh Rashid's 1955 summer majlis, a tiny coral, gypsum and timber "
        "retreat in a palm garden, restored and opened as a heritage house.",
    ),
}


# Extended editorial metadata used by the redesigned site-record modal. Where
# a Wikipedia page was mis-assigned in the dataset, these hand-written quick
# facts keep the record accurate without inventing dates or figures.
SITE_DETAILS = {
    "Al Fahidi Historical Neighbourhood": {
        "category": "HERITAGE DISTRICT",
        "arabicName": "حي الفهيدي التاريخي",
        "builtDate": "1890s",
        "location": "Al Fahidi, Bur Dubai",
        "materials": "Coral stone, gypsum, teak & sandalwood, palm",
        "era": "Late 19th century",
        "features": "Courtyard houses, wind towers (barjeel), narrow shaded lanes",
        "overview": (
            "Founded by affluent Persian merchants drawn from Bastak by Dubai's "
            "open trade, Al Fahidi is the city's best-preserved historic quarter. "
            "Its wind-tower courtyard houses were built in the late 1890s, and the "
            "quarter narrowly survived demolition in the 1980s after a preservation "
            "campaign led by British architect Rayner Otter. Today it is a living "
            "museum of Gulf architecture: lanes of coral and gypsum homes now hold "
            "galleries, the Coffee Museum, and the Sheikh Mohammed Centre for "
            "Cultural Understanding."
        ),
        "highlights": [
            "Wind towers (barjeel) rising above the courtyards",
            "Courtyard houses with carved teak doorways and gypsum panels",
            "The Sheikh Mohammed Centre for Cultural Understanding (SMCCU)",
            "The Coffee Museum, with live roasting and brewing demonstrations",
            "Street-level art galleries housed in the restored homes",
        ],
    },
    "Al Fahidi Fort": {
        "category": "FORT & MUSEUM",
        "arabicName": "حصن الفهيدي",
        "builtDate": "c. 1787",
        "location": "Al Fahidi, Bur Dubai",
        "materials": "Coral stone, gypsum, palm trunks",
        "era": "Late 18th century",
        "features": "Square stronghold, high watchtowers, crenellated walls, central courtyard",
        "overview": (
            "Al Fahidi Fort is Dubai's oldest surviving building, raised around "
            "1787 to guard the landward approaches to the settlement on the creek. "
            "Built from coral stone set in gypsum, its watchtower and thick walls "
            "kept watch over the city for two centuries. In 1971 the fort reopened "
            "as the Dubai Museum, its galleries winding beneath the courtyard to "
            "show dioramas of pearling, souk life, and the desert Bedouin, together "
            "with artefacts tracing the emirate back over 3,000 years."
        ),
        "highlights": [
            "The fort's watchtower, the oldest part of the building",
            "Coral and gypsum masonry in the walls and courtyard",
            "Underground galler with life-scale dioramas of old Dubai",
            "Pearling boats, swords and jewellery in the museum's collection",
            "The recreated spice souk with its rich, layered aromas",
        ],
    },
    "Heritage House": {
        "category": "HISTORIC HOUSE MUSEUM",
        "arabicName": "بيت التراث",
        "builtDate": "1890s",
        "location": "Al Ras, Deira",
        "materials": "Coral stone, gypsum, palm and mangrove",
        "era": "Late 19th century",
        "features": "Pearl-merchant courtyard house with wind tower and majlis",
        "overview": (
            "Built in the 1890s, Heritage House was the home of a wealthy pearl "
            "merchant who grew rich on the trade that once powered the Gulf. Set "
            "around a shady courtyard in Al Ras, the house shows how successful "
            "Deira merchants lived: a formal majlis for receiving guests, rooms "
            "lighted by carved gypsum grilles, and a wind tower drawing sea "
            "breezes down through the living quarters. Restored as a museum, it "
            "opens a window onto old family life in old Dubai."
        ),
        "highlights": [
            "The pearl merchant's formal majlis (reception room)",
            "Carved gypsum jali screens and ceiling decoration",
            "The courtyard and winding central stairwell",
            "Simple domestic furniture and household objects of the 1890s",
            "The wind tower that cooled the upper rooms",
        ],
    },
    "Al Ahmadiya School": {
        "category": "HERITAGE SCHOOL",
        "arabicName": "مدرسة الأحمدية",
        "builtDate": "1912",
        "location": "Al Ras, Deira",
        "materials": "Coral stone, gypsum, teak",
        "era": "Early 20th century",
        "features": "First formal school in Dubai with classrooms around a courtyard",
        "overview": (
            "Opened in 1912 by Sheikh Ahmed Bin Dalmouk, a pearl merchant, Al "
            "Ahmadiya was Dubai's first formal school. Its coral and gypsum rooms "
            "around a courtyard taught generations of pupils, before graduating "
            "many of the merchants and leaders who modernised twentieth-century "
            "Dubai. Restored in the mid-1990s and reopened as a museum in 2000, "
            "the building now charts the city's journey from informal Quranic "
            "study to a system of formal education."
        ),
        "highlights": [
            "Recreated classrooms with period desks and blackboards",
            "The coral and gypsum courtyard plan of the original school",
            "Displays on Dubai's early education system",
            "Portraits and stories of the school's notable alumni",
        ],
    },
    "Sheikh Saeed Al Maktoum House": {
        "category": "HERITAGE HOUSE & MUSEUM",
        "arabicName": "بيت الشيخ سعيد آل مكتوم",
        "builtDate": "1896",
        "location": "Al Shindagha, Bur Dubai",
        "materials": "Coral stone, gypsum, teak, palm",
        "era": "Late 19th century",
        "features": "Large creekside residence with two wind towers and nine museum wings",
        "overview": (
            "Stretching along the creek at Al Shindagha, this house was built "
            "around 1896 as the seat of the Al Maktoum family and home of Sheikh "
            "Saeed, Dubai's longest-serving ruler. Its long, low form is broken by "
            "two wind towers that kept the interiors cool, and its rooms open onto "
            "a courtyard facing the water. Now a museum, its nine wings display "
            "photographs of old Dubai, maps, marine life, coins and stamps, and the "
            "family history that shaped the city."
        ),
        "highlights": [
            "The two wind towers rising over the creek facade",
            "Historic photographs of old Dubai and its rulers",
            "The collections of coins, stamps and historic maps",
            "Displays on pearl diving and Dubai's marine life",
            "The long Shindagha waterfront setting along the creek",
        ],
    },
    "Al Shindagha Heritage District": {
        "category": "HERITAGE DISTRICT",
        "arabicName": "منطقة الشندغة التاريخية",
        "builtDate": "mid-1800s onwards",
        "location": "Al Shindagha, Bur Dubai",
        "materials": "Coral stone, gypsum, palm and mangrove",
        "era": "Mid-19th to early 20th century",
        "features": "Creek-front quarter of wind-tower houses, sikka lanes, a watchtower and museums",
        "overview": (
            "Al Shindagha is the historic creekside district where Dubai's ruling "
            "family and its merchants built coral-stone courtyard houses from the "
            "mid-19th century. Raised under the 'Dubai Historic District' "
            "restoration project, 162 buildings across 169,000 square metres have "
            "been brought back to life here. Winding sikka alleyways thread between "
            "wind-tower homes, the square Muraba'at Al Shindagha watchtower guards "
            "the district's tip, and the Al Shindagha Museum tells the story of the "
            "creek's birth of a city, all wrapped around the former home of the "
            "ruling family."
        ),
        "highlights": [
            "The Al Shindagha Museum and its 'Dubai Creek: Birth of a City' pavilion",
            "Sheikh Saeed Al Maktoum House, the family seat on the creek",
            "Wind-tower courtyard houses along the waterfront",
            "The narrow shaded sikka alleyways and hidden courtyards",
            "The Muraba'at watchtower rising at the mouth of the district",
        ],
    },
    "Dubai Creek": {
        "category": "NATURAL HERITAGE WATERWAY",
        "arabicName": "خور دبي",
        "builtDate": "natural waterway",
        "location": "Bur Dubai and Deira, Dubai",
        "materials": "Natural saltwater inlet",
        "era": "Heart of the city since ancient times",
        "features": "Saltwater creek dividing Deira and Bur Dubai with dhow wharves and abra crossings",
        "overview": (
            "Dubai Creek (Khor Dubai) is the saltwater inlet that divided old Dubai "
            "into Deira and Bur Dubai and made the city's fortune. For centuries, "
            "dhows loaded with pearls, spices, textiles and gold worked the creek, "
            "anchoring the pearling and trading economy on which the city was built. "
            "Its shores still hold the heritage quarters of Al Fahidi and Al "
            "Shindagha, the Deira souks, the dhow wharfage and the wooden abra boats "
            "that ferry passengers from one bank to the other for a dirham. Khor "
            "Dubai has been inscribed on ICESCO's List of Islamic World Tangible "
            "Heritage in recognition of that living history."
        ),
        "highlights": [
            "The traditional dhow wharfage where wooden trading dhows moor",
            "Abra rides across the creek between Bur Dubai and Deira",
            "The Spice and Gold Souks on the Deira shore",
            "Views of the Al Fahidi and Al Shindagha heritage districts",
            "The creek at sunset, lit by the towers of old and new Dubai",
        ],
    },
    "Jumeirah Mosque": {
        "category": "MOSQUE & CULTURAL LANDMARK",
        "arabicName": "مسجد جميرا",
        "builtDate": "1979",
        "location": "Jumeirah, Dubai",
        "materials": "White stone",
        "era": "Late 20th century, built in Fatimid style",
        "features": "Grand dome flanked by twin minarets with carved stone decoration",
        "overview": (
            "Built in Fatimid style between 1975 and 1979 as a gift from Sheikh "
            "Rashid bin Saeed Al Maktoum to his son Sheikh Mohammed, Jumeirah "
            "Mosque is one of Dubai's most photographed landmarks. Its white stone "
            "walls, grand dome and twin minarets are decorated with the intricate "
            "geometric carvings of medieval Cairo, and it can hold around 1,500 "
            "worshippers. Outside prayer times it runs the 'Open Doors. Open "
            "Minds.' programme, one of the few places in Dubai where non-Muslim "
            "visitors can join guided tours of Islam and Emirati culture."
        ),
        "highlights": [
            "The 'Open Doors. Open Minds.' guided tours by SMCCU",
            "Fatimid-style geometric carvings in white stone",
            "The grand central dome flanked by twin minarets",
            "Its landmark setting on Jumeirah Road near the beach",
            "Famous silhouettes of the mosque at sunset and at night",
        ],
    },
    "Bastakiya Mosque": {
        "category": "HISTORIC MOSQUE",
        "arabicName": "مسجد البستكية",
        "builtDate": "1890s",
        "location": "Al Fahidi Historical Neighbourhood, Bur Dubai",
        "materials": "Coral stone, gypsum, sandalwood",
        "era": "Late 19th century",
        "features": "Small neighbourhood mosque built with the Bastakiya quarter",
        "overview": (
            "Set in the lanes of Al Fahidi Historical Neighbourhood, once known as "
            "Al Bastakiya, this mosque is one of the quarter's oldest surviving "
            "buildings. It was raised by the Persian merchants from Bastak who "
            "founded the neighbourhood in the 1890s, built in the same coral stone "
            "and gypsum as the wind-tower houses around it. Small in scale and "
            "tightly woven into the urban fabric, with its slender minaret rising "
            "above the narrow sikka, the Bastakiya Mosque anchors the daily life of "
            "one of Dubai's most historic quarters."
        ),
        "highlights": [
            "Its coral and gypsum construction matching the old houses around it",
            "The slender minaret rising above the sikka lanes",
            "Its setting at the heart of the Al Fahidi quarter",
            "Quiet views across the wind-tower rooftops of Bastakiya",
            "Close to the Coffee Museum and SMCCU in Al Fahidi",
        ],
    },
    "Majlis Ghorfat Umm Al Sheif": {
        "category": "HISTORIC SUMMER MAJLIS",
        "arabicName": "مجلس غرفة أم الشيف",
        "builtDate": "1955",
        "location": "Jumeirah 2, Dubai",
        "materials": "Adobe, gypsum, coral and timber",
        "era": "Mid-20th century, restored 1994",
        "features": "Single-storey summer majlis with a roof terrace, palm garden and falaj irrigation",
        "overview": (
            "Established in 1955 as the summer retreat of the late Sheikh Rashid "
            "bin Saeed Al Maktoum, the Majlis Ghorfat Umm Al Sheif sits in the "
            "church-quiet Jumeirah area that was then a calm palm garden far from "
            "the bustle of old Dubai. Despite its small size, it is a special "
            "historical monument: Sheikh Rashid spent evenings here in solitude or "
            "conferring with his advisers. The flat roof terrace was used to dry "
            "dates and as a cool open-air sleeping platform. Fully restored in 1994 "
            "with a garden of date palms and a traditional aflaj irrigation system, "
            "the majlis now offers a rare glimpse of the historical furniture, tools "
            "and utensils of a traditional Emirati majlis — coffee stoves, "
            "earthenware, brassware, rugs and carpets, clocks, radios and the "
            "rifles and daggers men carried in their belts."
        ),
        "architecture": (
            "A single-storey summer house of adobe, gypsum and coral laid with "
            "timber, raised above the ground with a flat roof terrace that served "
            "both to dry dates and as a cool open-air sleeping platform in the hot "
            "Gulf summer. Interior rooms open directly onto the garden, and the "
            "original plan kept the interior quarters for living and the terrace "
            "for the family to sleep under the stars — cooled by the sea breeze "
            "that the open, palm-shaded setting let flow through the house."
        ),
        "highlights": [
            "The preserved 1955 summer majlis of Sheikh Rashid bin Saeed",
            "The roof terrace used for drying dates and open-air sleeping",
            "Historical furniture, copper, earthenware, brass and carpets",
            "Clocks and radios — the modern appliances of the 1950s",
            "The restored palm garden with traditional falaj irrigation",
        ],
    },
}


# Architecture & Design copy and official links from the Dubai Culture & Arts
# Authority (dubaiculture.gov.ae), merged into each site record by the loader.
SITE_ARCHITECTURE = {
    "Al Fahidi Fort": (
        "A square coral-and-gypsum stronghold of thick crenellated walls and a "
        "high corner watchtower, built from local reef stone set in gypsum mortar "
        "with palm-trunk roof timbers. Its one monumental gateway opens onto a "
        "wide courtyard, below which the Dubai Museum's galleries wind: a raised "
        "platform diorama of old Dubai's skyline leads through life-scale scenes "
        "of pearling, souk life and the desert."
    ),
    "Al Fahidi Historical Neighbourhood": (
        "Long-grown weather-bleached houses of stone, gypsum, teak, sandalwood, "
        "fronds and palm wood, pressed shoulder-to-shoulder and split by narrow "
        "shaded sikka alleys and small public squares. The signature element is "
        "the barjeel, or wind tower — a tall mud-brick shaft open to the breeze "
        "that catches the air above the rooftops and drives it down through the "
        "rooms. Houses wrap around shaded courtyards with carved teak doors and "
        "deep-set gypsum jali screens, each cross-ventilated by its own tower."
    ),
    "Sheikh Saeed Al Maktoum House": (
        "A long, low creekside residence of coral and gypsum masonry, its "
        "restrained facade broken by two tall wind towers that pulled the sea "
        "breeze through the rooms in summer. The double-height reception band "
        "opens onto a courtyard facing the water, with carved plaster, wood and "
        "sandalwood ceilings throughout. Nine wings and shaded verandas turn the "
        "building toward the flow of air and the view down the creek."
    ),
    "Al Ahmadiya School": (
        "A coral-and-gypsum school building ranged around a square central "
        "courtyard, with classrooms, storerooms and the headmaster's rooms on all "
        "four sides. Carved teak doors and window screens, wooden roof beams and "
        "deep plaster-lined walls kept interiors cool before air conditioning — "
        "the courtyard doing the work of a giant open-air classroom and play "
        "ground. The plan made light, shade and the wind-tower breeze do the work "
        "of modern climate control."
    ),
    "Heritage House": (
        "A pearl merchant's house of coral and gypsum arranged around an inner "
        "courtyard, all its rooms facing inward away from the street. A lofty "
        "wind tower above the court draws cool breezes down into the living "
        "quarters, while deep wooden verandas (diwans) shade the rooms and the "
        "carved gypsum jali screens filter the harsh midday light. Its compact, "
        "defensive, inward-looking plan is the classic form of old Gulf town "
        "houses."
    ),
    "Al Shindagha Heritage District": (
        "A creek-front quarter of coral-stone courtyard houses whose walls are "
        "washed in sand and lime, topped with tall barjeel wind towers that sieve "
        "the strongest breeze into the rooms below. Winding sikka alleyways thread "
        "between buildings, every house turned toward the water for trade and "
        "air, with wind towers, carved timber and gypsum detailing marking the "
        "wealthiest merchant and ruling-family homes along the waterfront."
    ),
    "Dubai Creek": (
        "Not a building but the saltwater inlet that shaped a city's layout: "
        "its banks divide Deira from Bur Dubai, its dhow wharfage, abra landings "
        "and the deep water that allowed centuries of pearling and trade. Its "
        "edges carry the old coral-stone quarters, wind-tower houses and souks, "
        "joined by wooden abras — a working waterfront rather than a monument."
    ),
    "Jumeirah Mosque": (
        "A luminous white-stone mosque built in Fatimid style between 1975 and "
        "1979, its grand central dome flanked by twin minarets. Walls, arches and "
        "the dome are dressed in the intricate carved geometric and floral "
        "patterns of medieval Cairo — a deliberate echo of Egypt's great Fatimid "
        "mosques. Sandstone-coloured by night lighting, it is one of Dubai's most "
        "photographed heritage silhouettes."
    ),
    "Bastakiya Mosque": (
        "A small neighbourhood mosque of the same coral stone and gypsum as the "
        "wind-tower houses around it, built in the 1890s as the quarter rose. Its "
        "slender minaret climbs above single-storey prayer hall, its low dome and "
        "plain, load-bearing walls woven tightly into the sikka lanes — a "
        "neighbourhood-scale building rather than a grand monument."
    ),
    "Majlis Ghorfat Umm Al Sheif": (
        "A single-storey summer house of adobe, gypsum and coral laid with "
        "timber, raised above the ground with a flat roof terrace that served "
        "both to dry dates and as a cool open-air sleeping platform in the hot "
        "Gulf summer. Interior rooms open directly onto the garden, and the "
        "original plan kept the living quarters shaded and breezy — cooled by the "
        "sea breeze that the open palm-shaded setting let flow through the house."
    ),
}

SITE_OFFICIAL_URLS = {
    "Al Fahidi Historical Neighbourhood": (
        "https://dubaiculture.gov.ae/en/attractions/heritage-sites/al-fahidi-historical-neighbourhood"
    ),
    "Heritage House": "https://dubaiculture.gov.ae/en/attractions/heritage-sites/heritage-house",
    "Al Ahmadiya School": "https://dubaiculture.gov.ae/en/attractions/heritage-sites/al-ahmadiya-school",
    "Majlis Ghorfat Umm Al Sheif": (
        "https://dubaiculture.gov.ae/en/attractions/heritage-sites/majlis-ghorfat-umm-al-sheif"
    ),
    "Al Fahidi Fort": "https://dubaiculture.gov.ae/en",
    "Al Shindagha Heritage District": "https://alshindagha.dubaiculture.gov.ae/en",
    "Sheikh Saeed Al Maktoum House": "https://dubaiculture.gov.ae/en",
}


def slugify(name):
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


@st.cache_data
def load_heritage_sites_from_json():
    """Loads dubai_buildings_full.json into the shape the views consume."""
    if not DATA_FILE.exists():
        st.error(f"Dataset not found at {DATA_FILE}.")
        return []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_sites = []
    for site in raw_data:
        name = site["site_name"]
        wiki = site.get("wikipedia") or {}
        images = site.get("images") or []
        summary = wiki.get("summary", "No description available.")

        # Sites missing from SITE_CARDS fall back to the Wikipedia first sentence
        district, blurb = SITE_CARDS.get(
            name, ("Dubai", summary.split(". ")[0] + ".")
        )

        details = SITE_DETAILS.get(name, {})
        gallery = [
            {
                "image_url": img["image_url"],
                "title": img.get("title", ""),
                "file_page_url": img.get("file_page_url", ""),
                "artist": img.get("artist", "Wikimedia Commons"),
            }
            for img in images[:3]
            if img.get("image_url")
        ]

        cleaned_sites.append(
            {
                "id": slugify(name),
                "name": name,
                "district": district,
                "blurb": blurb,
                "lat": site.get("lat"),
                "lon": site.get("lon"),
                "image": images[0]["image_url"] if images else None,
                "summary": summary,
                "category": details.get("category", "HERITAGE SITE"),
                "arabicName": details.get("arabicName", ""),
                "builtDate": details.get("builtDate", ""),
                "location": details.get("location", district),
                "materials": details.get("materials", "Traditional Materials"),
                "era": details.get("era", ""),
                "features": details.get("features", "Wind Tower Architecture"),
                "overview": details.get(
                    "overview", summary or "No description available."
                ),
                "architecture": SITE_ARCHITECTURE.get(name, ""),
                "officialUrl": SITE_OFFICIAL_URLS.get(name, ""),
                "gallery": gallery,
                "highlights": details.get("highlights", []),
            }
        )

    return cleaned_sites


SITES = load_heritage_sites_from_json()


def get_site_by_id(site_id):
    for s in SITES:
        if str(s["id"]) == str(site_id):
            return s
    return None


# ==============================================================================
# 3. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Athar, Digital Archive of Dubai",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #F8F5EE;
        color: #2C221E;
        font-family: 'Karla', sans-serif;
    }
    header, footer { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    /* Main-page header (mirrors the reference Sikka layout) */
    .main-header {
        text-align: center;
        max-width: 52rem;
        margin: 0 auto 1.6rem;
    }
    .main-header-tag {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #B3532D;
        margin-bottom: 0.9rem;
    }
    .main-header-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 3.6rem;
        font-weight: 700;
        line-height: 1.05;
        color: #231610;
        margin-bottom: 0.7rem;
    }
    .main-title-dot { color: #C8972C; }
    .main-header-sub {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #6E6056;
        margin: 0 auto;
        max-width: 40rem;
    }
    .landing-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 4.2rem;
        font-weight: 700;
        text-align: center;
        color: #231610;
        line-height: 1.1;
        margin-bottom: 2rem;
    }

    /* ------------------------------------------------------------------
       LANDING — clean centered welcome (no background photo)
    ------------------------------------------------------------------ */
    .landing-body {
        text-align: center;
        max-width: 54rem;
        margin: 0 auto;
        padding: 3.5rem 1rem 1.4rem;
    }
    .landing-tag {
        display: inline-block;
        letter-spacing: 3px;
        font-size: 0.78rem;
        font-weight: 800;
        color: #B3532D;
        text-transform: uppercase;
        margin-bottom: 1.3rem;
    }
    .landing-body .landing-title {
        font-size: 4.2rem;
        color: #231610;
        margin-bottom: 1.1rem;
    }
    .landing-sub {
        font-size: 1.12rem;
        line-height: 1.65;
        color: #6E6056;
        font-weight: 400;
        max-width: 40rem;
        margin: 0 auto 0.4rem;
    }
    .st-key-landing_cta {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 2.2rem 0 0;
    }
    .st-key-landing_go {
        display: flex;
        justify-content: center;
    }
    .st-key-landing_go [data-testid="stButton"] {
        width: auto !important;
    }
    .st-key-landing_go [data-testid="stButton"] button {
        background: #B3532D !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        padding: 0.85rem 3.2rem !important;
        font-size: 1.08rem !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    .st-key-landing_go [data-testid="stButton"] button:hover {
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22) !important;
    }
    .landing-note {
        text-align: center;
        font-size: 0.82rem;
        color: #9A8A7C;
        margin: 1.4rem 0 0;
    }

    /* ------------------------------------------------------------------
       ARCHIVE CARDS — parchment tiles (mirrors the reference "In the archive")
    ------------------------------------------------------------------ */
    .site-card {
        background: #FFFFFF;
        border: 1px solid #E5DCC9;
        border-radius: 18px;
        padding: 0.9rem;
        box-shadow: 0 1px 2px rgba(44, 34, 30, 0.05), 0 12px 30px rgba(44, 34, 30, 0.07);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .site-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 10px rgba(44, 34, 30, 0.06), 0 18px 40px rgba(44, 34, 30, 0.10);
    }
    .site-card-img {
        width: 100%;
        aspect-ratio: 3 / 2;
        object-fit: cover;
        object-position: center;
        border-radius: 12px;
        display: block;
        background-color: #EFE8DA;
    }
    .site-card-meta {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #8C7B70;
        margin: 0.7rem 0 0;
    }
    .site-card-meta b {
        color: #B3532D;
        font-weight: 800;
    }
    .site-card-name {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #231610;
        line-height: 1.3;
        margin: 0.2rem 0 0;
        min-height: 2.6em;
    }
    .site-card-blurb {
        font-size: 0.85rem;
        color: #6E6056;
        line-height: 1.45;
        margin: 0.3rem 0 0;
        min-height: 2.9em;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    div[class*="st-key-card_btn_"] {
        margin-top: auto;
    }
    div[class*="st-key-card_btn_"] [data-testid="stButton"] button {
        background: transparent !important;
        color: #B3532D !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.45rem 0 0.15rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        width: 100% !important;
        text-align: left !important;
        box-shadow: none !important;
        margin-top: 0.3rem;
        transition: color 0.15s ease !important;
    }
    div[class*="st-key-card_btn_"] [data-testid="stButton"] button:hover {
        color: #9A6A44 !important;
    }

    /* ------------------------------------------------------------------
       "ASK THE ARCHIVE ANYTHING" BAND (mirrors the reference chat section)
    ------------------------------------------------------------------ */
    .st-key-chat_band {
        background: #F5EEDF;
        border: 1px solid #E8DCC6;
        border-top: 3px solid #B3532D;
        border-radius: 24px;
        padding: 2.2rem 2.5rem 1.6rem;
        margin: 2rem auto 2.4rem;
        width: 100%;
        max-width: none;
    }
    .chat-band-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: #231610;
        margin: 0;
    }
    .chat-band-sub {
        font-size: 0.92rem;
        color: #6E6056;
        line-height: 1.55;
        margin: 0.4rem 0 1rem;
        max-width: 40rem;
    }

    /* Prompt suggestion chips */
    .st-key-chip_0, .st-key-chip_1, .st-key-chip_2 {
        display: flex;
        align-items: center;
        justify-content: center;
        align-self: center;
    }
    .st-key-chip_0 [data-testid="stButton"] button,
    .st-key-chip_1 [data-testid="stButton"] button,
    .st-key-chip_2 [data-testid="stButton"] button {
        background: #FFFFFF !important;
        color: #6E6056 !important;
        border: 1px solid #E0D8C8 !important;
        border-radius: 999px !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        white-space: nowrap !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
    .st-key-chip_0 [data-testid="stButton"] button:hover,
    .st-key-chip_1 [data-testid="stButton"] button:hover,
    .st-key-chip_2 [data-testid="stButton"] button:hover {
        color: #B3532D !important;
        border-color: #B3532D !important;
    }

    /* Pill input + circular send button */
    .st-key-chat_askbar {
        position: relative;
        display: flex;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E0D8C8;
        border-radius: 999px;
        padding: 0.1rem 0.2rem 0.1rem 1.1rem;
        box-shadow: 0 1px 4px rgba(44, 34, 30, 0.06);
        margin-top: 0.6rem;
    }
    .st-key-chat_askbar::before {
        content: "✦";
        position: absolute;
        left: 1.05rem;
        top: 50%;
        transform: translateY(-50%);
        z-index: 3;
        font-size: 0.95rem;
        color: #C8972C;
        pointer-events: none;
    }
    .st-key-chat_askbar [data-testid="stTextInput"] {
        background: transparent;
    }
    .st-key-chat_askbar [data-testid="stTextInput"] input {
        border: none !important;
        box-shadow: none !important;
        background: transparent;
        padding-left: 1.9rem;
    }
    .st-key-chat_askbar [data-testid="stTextInput"] div {
        background: transparent;
    }
    .st-key-chat_askbar [data-testid="stTextInput"] label {
        display: none;
    }
    .st-key-chat_askbar > div [data-testid="stButton"] button {
        background: #B3532D !important;
        color: #FFFFFF !important;
        border: none !important;
        width: 2.6rem;
        height: 2.6rem;
        min-width: 2.6rem !important;
        padding: 0 !important;
        border-radius: 50% !important;
        font-size: 0.9rem;
        box-shadow: none !important;
    }
    .st-key-chat_askbar > div [data-testid="stButton"] button:hover {
        background: #9A6A44 !important;
    }
    .st-key-chat_askbar [data-testid="stButton"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Small "Back to Welcome" pill at the top of the archive page */
    .st-key-back_link [data-testid="stButton"] button {
        background: #FFFFFF00 !important;
        color: #8C7B70 !important;
        border: 1px solid #E0D8C8 !important;
        border-radius: 999px !important;
        padding: 0.35rem 1rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        width: auto !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
.st-key-back_link [data-testid="stButton"] button:hover {
        color: #B3532D !important;
        border-color: #B3532D !important;
    }

    /* Sidebar chat library — conversation pills + delete button */
    div[class*="st-key-load_conv_"] [data-testid="stButton"] button {
        background: #FBF7F0 !important;
        color: #3A2E27 !important;
        border: 1px solid #E8DCC6 !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.6rem !important;
        font-size: 0.8rem !important;
        line-height: 1.35 !important;
        text-align: left !important;
        box-shadow: none !important;
        white-space: normal !important;
        justify-content: flex-start !important;
        transition: border-color 0.15s ease !important;
    }
    div[class*="st-key-load_conv_"] [data-testid="stButton"] button:hover {
        border-color: #B3532D !important;
    }
    div[class*="st-key-del_conv_"] {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: auto;
        margin-bottom: auto;
    }
    div[class*="st-key-del_conv_"] [data-testid="stButton"] button {
        background: #FBF3EC !important;
        color: #B3532D !important;
        border: 1px solid #EED9C4 !important;
        border-radius: 8px !important;
        padding: 0.2rem 0.5rem !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        box-shadow: none !important;
        width: auto !important;
        line-height: 1 !important;
        transition: background 0.15s ease !important;
    }
    div[class*="st-key-del_conv_"] [data-testid="stButton"] button:hover {
        background: #FBE8D8 !important;
    }

    /* Footer note */
    .archive-footer {
        text-align: center;
        font-size: 0.8rem;
        color: #9A8A7C;
        margin: 1.6rem 0 0.4rem;
    }

    /* General chat panel + new-chat header row */
    .st-key-chat_head {
        margin-bottom: 0.6rem;
    }
    .st-key-chat_head h3 {
        margin: 0;
    }
    .st-key-new_chat {
        display: flex;
        justify-content: flex-end;
    }
    .st-key-chat_head [data-testid="stButton"] button {
        background: #FFFFFF !important;
        color: #B3532D !important;
        border: 1px solid #D9B48F !important;
        border-radius: 999px !important;
        padding: 0.45rem 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: background 0.15s ease !important;
    }
    .st-key-chat_head [data-testid="stButton"] button:hover {
        background: #FBF6EE !important;
    }
    .st-key-photo_bar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .st-key-photo_bar [data-testid="stFileUploader"] {
        flex: 1;
    }
    .st-key-photo_go {
        display: flex;
        align-items: center;
        margin-top: auto;
        margin-bottom: auto;
    }
    .st-key-photo_bar [data-testid="stFileUploader"] section {
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        border: 1px dashed #D9B48F;
        background: #FFFFFF;
    }
    .st-key-photo_bar [data-testid="stFileUploader"] button {
        border-radius: 999px !important;
    }
    .photo-bubble-img {
        width: 100%;
        max-width: 220px;
        aspect-ratio: 4 / 3;
        object-fit: cover;
        border-radius: 14px;
        display: block;
        margin-top: 0.2rem;
        box-shadow: 0 2px 10px rgba(44, 34, 30, 0.12);
    }
    .match-card {
        display: flex;
        gap: 1rem;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E4D6C4;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin: 0.4rem 0 0.9rem;
        box-shadow: 0 2px 10px rgba(44, 34, 30, 0.07);
    }
    .match-card-img {
        width: 92px;
        height: 92px;
        flex-shrink: 0;
        object-fit: cover;
        object-position: center;
        border-radius: 12px;
        display: block;
    }
    .match-card-info { min-width: 0; }
    .match-card-kicker {
        font-size: 0.68rem;
        font-weight: 800;
        color: #B3532D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .match-card-name {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.12rem;
        font-weight: 700;
        color: #231610;
        line-height: 1.25;
    }
    .match-card-blurb {
        font-size: 0.83rem;
        color: #6E6056;
        line-height: 1.4;
    }
    .meta-pill-box {
        background-color: #EFE8DA;
        border-radius: 10px;
        padding: 0.8rem;
        margin-top: 0.5rem;
    }
    .meta-pill-title {
        font-size: 0.7rem;
        font-weight: 800;
        color: #8C7B70;
        text-transform: uppercase;
    }
    .meta-pill-value {
        font-size: 0.85rem;
        font-weight: 600;
        color: #231610;
    }
    div.stButton > button {
        background-color: #B3532D !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
    }
    div.stButton > button:hover {
        background-color: #9A6A44 !important;
    }
    .site-square-empty {
        background-color: #EFE8DA;
        border: 1px solid #E0D8C8;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1.2rem;
        box-sizing: border-box;
        aspect-ratio: 1 / 1;
        width: 100%;
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #B3532D;
    }
    .user-bubble {
        background-color: #B3532D;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        float: right;
        margin-bottom: 0.8rem;
        max-width: 75%;
        font-size: 0.95rem;
    }
    .assistant-bubble {
        background-color: #EFE8DA;
        color: #231610;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        float: left;
        margin-bottom: 1rem;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .source-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.7rem;
        clear: both;
    }
    .source-pill {
        display: inline-block;
        background: #F3ECDF;
        border: 1px solid #D9B48F;
        color: #B3532D;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.74rem;
        line-height: 1.5;
        text-decoration: none;
    }
    .source-pill:hover {
        background: #B3532D;
        color: #FFFFFF;
        text-decoration: none;
    }

    /* ------------------------------------------------------------------
       EDITORIAL SITE-RECORD MODAL
    ------------------------------------------------------------------ */

    /* Hero banner with a real image layer and a dark gradient overlay */
    .site-hero {
        position: relative;
        width: 100%;
        aspect-ratio: 16 / 9;
        min-height: 14rem;
        border-radius: 18px;
        overflow: hidden;
        background-color: #6B4E3D;
        background-position: center;
        background-size: cover;
        background-repeat: no-repeat;
        margin-bottom: 1rem;
    }
    .site-hero-overlay {
        position: absolute;
        inset: 0;
        z-index: 2;
        display: flex;
        align-items: flex-end;
        background: linear-gradient(
            to top,
            rgba(16, 10, 6, 0.92) 0%,
            rgba(16, 10, 6, 0.55) 45%,
            rgba(16, 10, 6, 0.05) 75%
        );
    }
    .site-hero-content {
        padding: 1.6rem 1.8rem 1.4rem;
        color: #FFFFFF;
    }
    .site-hero-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #F2C29B;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.55rem;
    }
    .site-hero-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2.15rem;
        font-weight: 700;
        line-height: 1.05;
        margin: 0;
        text-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
    }
    .site-hero-arabic {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.2rem;
        color: rgba(255, 255, 255, 0.95);
        text-shadow: 0 1px 8px rgba(0, 0, 0, 0.35);
    }
    .site-hero-blurb {
        font-size: 0.95rem;
        line-height: 1.45;
        margin-top: 0.6rem;
        color: rgba(255, 255, 255, 0.92);
        max-width: 46rem;
        text-shadow: 0 1px 6px rgba(0, 0, 0, 0.4);
    }

    /* Metadata bar just below the hero */
    .site-meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        align-items: center;
        background: #EFE8DA;
        border-radius: 12px;
        padding: 0.85rem 1.2rem;
        margin-bottom: 1.6rem;
    }
    .site-meta-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #6E6056;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .site-meta-item svg, .site-meta-item img {
        flex-shrink: 0;
        color: #B3532D;
    }
    .site-meta-pill {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid #D9B48F;
        color: #B3532D;
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: auto;
        text-decoration: none;
        transition: background 0.15s ease;
    }
    .site-meta-pill:hover {
        background: #B3532D;
        color: #FFFFFF;
    }

    /* Section heading */
    .site-section-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #231610;
        margin: 1.6rem 0 0.6rem;
    }
    .site-kicker-label {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #B3532D;
        margin-bottom: 0.2rem;
    }

    /* Overview body paragraph */
    .site-overview {
        font-size: 1rem;
        line-height: 1.7;
        color: #3A2E28;
        max-width: 46rem;
        margin: 0;
    }
    .site-architecture {
        font-size: 1rem;
        line-height: 1.7;
        color: #3A2E28;
        max-width: 46rem;
        margin: 0.4rem 0 1rem;
        padding: 1.1rem 1.3rem;
        background: #F0E7DB;
        border-left: 3px solid #B3532D;
        border-radius: 0 12px 12px 0;
    }

    /* 3-column quick-fact spec cards */
    .spec-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .spec-card {
        flex: 1;
        background: #EFE8DA;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        min-width: 0;
    }
    .spec-icon {
        font-size: 1.3rem;
        margin-bottom: 0.45rem;
        display: block;
    }
    .spec-label {
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8C7B70;
        margin-bottom: 0.3rem;
    }
    .spec-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #231610;
        line-height: 1.4;
    }

    /* 3-column image gallery grid */
    .gallery-row {
        display: flex;
        gap: 0.9rem;
    }
    .gallery-cell {
        flex: 1;
        min-width: 0;
    }
    .gallery-img {
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        object-position: center;
        background-color: #EFE8DA;
        border-radius: 12px;
        display: block;
    }
    .gallery-cap {
        font-size: 0.74rem;
        color: #8C7B70;
        line-height: 1.35;
        margin-top: 0.4rem;
    }

    /* Highlights checklist */
    .highlight-list {
        list-style: none;
        margin: 0;
        padding: 0;
    }
    .highlight-list li {
        position: relative;
        padding-left: 1.6rem;
        margin-bottom: 0.55rem;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #3A2E28;
    }
    .highlight-list li::before {
        content: "✓";
        position: absolute;
        left: 0;
        top: 0;
        color: #B3532D;
        font-weight: 800;
    }

    /* Interactive AI assistant card */
    .ai-card {
        background: #F0E7DB;
        border: 1px solid #E4D6C4;
        border-radius: 18px;
        padding: 1.5rem 1.5rem 0.2rem;
        margin-top: 2rem;
    }
    .ai-card-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #231610;
        margin: 0;
    }
    .ai-card-sub {
        font-size: 0.9rem;
        color: #6E6056;
        margin: 0.25rem 0 1.1rem;
    }
    .ai-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 1.1rem;
    }
    .ai-pill {
        background: #FFFFFF;
        border: 1px solid #E0D8C8;
        color: #B3532D;
        border-radius: 999px;
        padding: 0.45rem 1rem;
        font-size: 0.83rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    .ai-pill:hover {
        background: #FBF6EE;
    }

    /* Inline search/chat input pill with leading icon + circular send button */
    .st-key-askbar {
        position: relative;
        display: flex;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E0D8C8;
        border-radius: 999px;
        padding: 0.1rem 0.2rem 0.1rem 1rem;
        box-shadow: 0 1px 4px rgba(44, 34, 30, 0.06);
    }
    .st-key-askbar::before {
        content: "🔍";
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        z-index: 3;
        font-size: 0.95rem;
        pointer-events: none;
    }
    .st-key-askbar [data-testid="stTextInput"] {
        background: transparent;
    }
    .st-key-askbar [data-testid="stTextInput"] input {
        border: none !important;
        box-shadow: none !important;
        background: transparent;
        padding-left: 1.9rem;
    }
    .st-key-askbar [data-testid="stTextInput"] div {
        background: transparent;
    }
    .st-key-askbar [data-testid="stTextInput"] label {
        display: none;
    }
    .st-key-askbar > div [data-testid="stButton"] button {
        background: #B3532D !important;
        color: #FFFFFF !important;
        border: none !important;
        width: 2.6rem;
        height: 2.6rem;
        min-width: 2.6rem !important;
        padding: 0 !important;
        border-radius: 50% !important;
        font-size: 1rem;
        box-shadow: none !important;
    }
    .st-key-askbar > div [data-testid="stButton"] button:hover {
        background: #9A6A44 !important;
    }

    /* Quick-suggestion pills rendered as Streamlit buttons */
    .st-key-askbar [data-testid="stButton"] {
        display: flex;
        align-items: center;
    }
    /* All dialog buttons default to white pill style; the send button inside
       .st-key-askbar overrides this with its circular brand styling. */
    div[data-testid="stDialog"] [data-testid="stButton"] > button {
        background: #FFFFFF !important;
        color: #B3532D !important;
        border: 1px solid #E0D8C8 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    div[data-testid="stDialog"] [data-testid="stButton"] > button:hover {
        background: #FBF6EE !important;
    }

    /* Sidebar themed to match the warm archive palette */
    section[data-testid="stSidebar"] {
        background-color: #F3ECDF;
    }
    section[data-testid="stSidebar"] h3 {
        color: #2C221E;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #8C7B70;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #E0D6C2;
    }
    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="stExpandSidebarButton"]:hover {
        color: #B3532D;
    }

    /* The only reopen control for a collapsed sidebar lives in the hidden top
       bar (stExpandSidebarButton inside stHeader). Show just that button as a
       small terracotta circle at the top-left so the chat library can always
       be reopened after collapsing. */
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
        visibility: visible;
        margin: 0.5rem;
    }
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] button {
        background: #B3532D !important;
        color: #FFFFFF !important;
        border-radius: 999px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 4. SESSION STATE
# ==============================================================================
if "view" not in st.session_state:
    st.session_state.view = "landing"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "site_chat" not in st.session_state:
    st.session_state.site_chat = {}

if "active_site_id" not in st.session_state:
    st.session_state.active_site_id = None

if "last_archive_question" not in st.session_state:
    st.session_state.last_archive_question = ""


# ==============================================================================
# 5. CHAT PERSISTENCE + SIDEBAR CHAT LIBRARY (mirrors the AI Study Buddy app)
# ==============================================================================
CONV_DIR = Path(__file__).parent / "conversations"
CONV_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _file_mime(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")


def _save_photo(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix or ".jpg"
    stem = datetime.datetime.now().strftime("%H%M%S_%f")
    dest = UPLOAD_DIR / f"photo_{stem}{ext}"
    dest.write_bytes(uploaded_file.getvalue())
    return str(dest)


def _run_photo_search(uploaded_file) -> None:
    """Identify an uploaded photo's archive site and log the exchange."""
    if uploaded_file is None:
        return
    file_bytes = uploaded_file.getvalue()
    save_path = _save_photo(uploaded_file)
    st.session_state.chat_history.append(
        {"role": "user", "content": "Search the archive by photo", "image_path": save_path}
    )
    with st.spinner("Reading your photo…"):
        site, err = identify_site_from_image(file_bytes, _file_mime(save_path))

    if site:
        ans, srcs = query_rag_safe(
            f"Tell me about {site['name']}, its history and architecture.",
            site_id=site["id"],
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": (
                    f"That looks like **{site['name']}** — {site['blurb']}\n\n{ans}"
                ),
                "sources": srcs,
                "match_site_id": site["id"],
            }
        )
    elif err:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": err, "sources": []}
        )
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": (
                    "That photo doesn't clearly match any record in the archive yet. "
                    "Try a shot of one of our heritage buildings — or ask about wind "
                    "towers, coral houses, pearling or our heritage sites."
                ),
                "sources": [],
            }
        )
    save_general_chat()
    st.rerun()


def _render_general_messages(messages: list[dict]) -> None:
    """Render general-archive chat bubbles, including photo messages and
    site-match cards."""
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>{msg['content']}</div><div style='clear: both;'></div>",
                unsafe_allow_html=True,
            )
            img_path = msg.get("image_path")
            if img_path and Path(img_path).exists():
                encoded = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
                mime = _file_mime(img_path)
                st.markdown(
                    f"<img class='photo-bubble-img' src='data:{mime};base64,{encoded}'>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div class='assistant-bubble'>{_assistant_html(msg.get('content', ''), msg.get('sources'))}</div><div style='clear: both;'></div>",
                unsafe_allow_html=True,
            )
            match_id = msg.get("match_site_id")
            if match_id:
                site = get_site_by_id(match_id)
                if site:
                    name = html.escape(site["name"])
                    st.markdown(
                        f"<div class='match-card'>"
                        f"<img class='match-card-img' src='{html.escape(site['image'], quote=True)}' alt='{name}'>"
                        f"<div class='match-card-info'>"
                        f"<div class='match-card-kicker'>{html.escape(site['district']).upper()}</div>"
                        f"<div class='match-card-name'>{name}</div>"
                        f"<div class='match-card-blurb'>{html.escape(site['blurb'])}</div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "View Site Record",
                        key=f"match_btn_{match_id}_{abs(hash(msg['content']))}",
                        width="stretch",
                    ):
                        st.session_state.active_site_id = site["id"]
                        st.rerun()


def _chat_title(messages: list[dict]) -> str:
    """First user message, truncated, used as the saved-chat title."""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
            return (text[:35] + "…") if len(text) > 35 else text
    return "Untitled chat"


def _save_conv(stem: str, payload: dict) -> None:
    (CONV_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )


def save_site_chat(site_id: str, site_name: str) -> str:
    """Write the current site chat to a file and return its stem."""
    messages = st.session_state.site_chat.get(site_id, [])
    stem = st.session_state.get("site_conv_stem", {}).get(
        site_id, f"{slugify(site_name)}_{datetime.datetime.now().strftime('%H%M%S')}"
    )
    st.session_state.site_conv_stem = st.session_state.setdefault(
        "site_conv_stem", {}
    )
    st.session_state.site_conv_stem[site_id] = stem
    _save_conv(
        stem,
        {
            "kind": "site",
            "site_id": site_id,
            "site_name": site_name,
            "messages": messages,
        },
    )
    return stem


def save_general_chat() -> str:
    """Write the general archive chat to a file and return its stem."""
    messages = st.session_state.chat_history
    stem = st.session_state.get(
        "general_conv_stem",
        f"archive_{datetime.datetime.now().strftime('%H%M%S')}",
    )
    st.session_state.general_conv_stem = stem
    _save_conv(
        stem,
        {
            "kind": "general",
            "site_name": "Archive",
            "messages": messages,
        },
    )
    return stem


def list_convs() -> list[Path]:
    return sorted(CONV_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_conv(stem: str) -> dict:
    return json.loads((CONV_DIR / f"{stem}.json").read_text())


def delete_conv(stem: str) -> None:
    (CONV_DIR / f"{stem}.json").unlink(missing_ok=True)


def answer_general_question(question: str) -> None:
    """Ask the general archive curator and persist the exchange."""
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Consulting local archive pipeline…"):
        answer, sources = query_rag_safe(question)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
    save_general_chat()
    st.rerun()


def render_sidebar_chat_library() -> None:
    """Sidebar with the saved-chat list (load / delete) for the main view."""
    with st.sidebar:
        st.markdown("### 💬 Chat Library")
        st.caption("Your saved conversations with Athar.")

        saved = list_convs()
        if saved:
            for conv_file in saved:
                payload = None
                try:
                    payload = load_conv(conv_file.stem)
                except Exception:
                    continue
                label = payload.get("site_name", conv_file.stem)
                title = _chat_title(payload.get("messages", []))
                col_load, col_del = st.columns([5, 1])
                with col_load:
                    if st.button(
                        f"{label} · {title}",
                        key=f"load_conv_{conv_file.stem}",
                        width="stretch",
                    ):
                        if payload.get("kind") == "site":
                            sid = payload.get("site_id")
                            st.session_state.site_chat[sid] = payload["messages"]
                            st.session_state.site_conv_stem = st.session_state.setdefault(
                                "site_conv_stem", {}
                            )
                            st.session_state.site_conv_stem[sid] = conv_file.stem
                            st.session_state.active_site_id = sid
                        else:
                            st.session_state.chat_history = payload["messages"]
                            st.session_state.general_conv_stem = conv_file.stem
                        st.rerun()
                with col_del:
                    if st.button(
                        "✕",
                        key=f"del_conv_{conv_file.stem}",
                        help="Delete this saved chat",
                    ):
                        delete_conv(conv_file.stem)
                        st.rerun()
        else:
            st.caption("No saved chats yet. Ask a question to start one.")

        st.sidebar.markdown("---")
        if st.sidebar.button("Open the Archive", key="to_archive", width="stretch"):
            st.session_state.view = "main"
            st.session_state.active_site_id = None
            st.rerun()


# ==============================================================================
# 5. SITE DIALOG POPUP
# ==============================================================================
def _assistant_html(content, sources=None):
    """Bubble content for assistant replies: escaped text, rendered **bold**,
    and Perplexity-style source pills (in a row below the answer)."""
    text = html.escape(content or "")

    # Render **...** as real bold.
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)

    pills = []
    seen = set()

    def _add_pill(label, url):
        key = url.rstrip("/")
        if key in seen:
            return
        seen.add(key)
        pills.append((html.escape(label), html.escape(url, quote=True)))

    for s in sources or []:
        url = s.get("url")
        if url:
            label = s.get("label") or url
            _add_pill(label, url)

    # Web-fallback replies carry "Source: <url>" lines — turn them into pills.
    for url in re.findall(r"(?im)^Source:\s*(https?://\S+)\s*$", content or ""):
        label = url.split("//")[-1].rstrip("/")
        _add_pill(label, url)
    text = re.sub(r"(?im)^Source:\s*(https?://\S+)\s*$", "", text)

    if pills:
        pill_html = "".join(
            f"<a class='source-pill' href='{url}' target='_blank' rel='noopener noreferrer'>{label}</a>"
            for label, url in pills
        )
        text = f"{text}<div class='source-pill-row'>{pill_html}</div>"

    return text


def _pill_ask(site, question, answer, sources=None):
    """Record a Q&A, persist it, and let the natural rerun keep the dialog open."""
    site_id = site["id"]
    chat = st.session_state.site_chat.setdefault(site_id, [])
    chat.append({"role": "user", "content": question})
    chat.append(
        {"role": "assistant", "content": answer, "sources": sources or []}
    )
    save_site_chat(site_id, site["name"])


@st.dialog("Site Record", width="large")
def render_site_dialog(site):
    site_id = site["id"]

    if st.button("← Back to Map", key=f"back_{site_id}", width="stretch"):
        st.session_state.active_site_id = None
        st.rerun()

    chat = st.session_state.site_chat.setdefault(site_id, [])

    esc = html.escape
    name = esc(site["name"])
    arabic = esc(site["arabicName"])
    category = esc(site["category"])
    blurb = esc(site["blurb"])
    built = esc(site["builtDate"])
    location = esc(site["location"])
    overview = esc(site["overview"])
    hero_image = site["image"]

    # --- HERO BANNER -------------------------------------------------------
    hero_bg = ""
    if hero_image:
        hero_bg = f" style='background-image: url(\"{esc(hero_image, quote=True)}\");'"

    st.markdown(
        f"""
        <div class='site-hero'{hero_bg}>
            <div class='site-hero-overlay'>
                <div class='site-hero-content'>
                    <span class='site-hero-tag'>{category}</span>
                    <h2 class='site-hero-title'>{name}</h2>
                    {f"<div class='site-hero-arabic'>{arabic}</div>" if arabic else ""}
                    <div class='site-hero-blurb'>{blurb}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- METADATA BAR ------------------------------------------------------
    meta_items = []
    if location:
        meta_items.append(
            f"<div class='site-meta-item'><svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0Z'/><circle cx='12' cy='10' r='3'/></svg>{location}</div>"
        )
    if built:
        meta_items.append(
            f"<div class='site-meta-item'><svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='4' width='18' height='18' rx='2'/><line x1='16' y1='2' x2='16' y2='6'/><line x1='8' y1='2' x2='8' y2='6'/><line x1='3' y1='10' x2='21' y2='10'/></svg>Built c. {built}</div>"
        )
    if site.get("officialUrl"):
        meta_items.append(
            f"<a class='site-meta-pill' href='{esc(site['officialUrl'], quote=True)}' target='_blank' rel='noopener noreferrer'>Official · Dubai Culture ↗</a>"
        )
    st.markdown(
        f"<div class='site-meta-bar'>{''.join(meta_items)}</div>",
        unsafe_allow_html=True,
    )

    # --- OVERVIEW & KEY SPEC CARDS ----------------------------------------
    st.markdown(
        "<div class='site-section-title'>Overview</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p class='site-overview'>{overview}</p>",
        unsafe_allow_html=True,
    )

    # --- ARCHITECTURE & DESIGN ---------------------------------------------
    architecture = site.get("architecture", "")
    if architecture:
        st.markdown(
            "<div class='site-section-title'>Architecture & Design</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='site-architecture'>{esc(architecture)}</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='spec-row'>
            <div class='spec-card'>
                <span class='spec-icon'>🧱</span>
                <div class='spec-label'>Materials</div>
                <div class='spec-value'>__MATERIALS__</div>
            </div>
            <div class='spec-card'>
                <span class='spec-icon'>🏛️</span>
                <div class='spec-label'>Historical Era / Key Date</div>
                <div class='spec-value'>__ERA__</div>
            </div>
            <div class='spec-card'>
                <span class='spec-icon'>🪟</span>
                <div class='spec-label'>Architectural Features</div>
                <div class='spec-value'>__FEATURES__</div>
            </div>
        </div>
        """.replace(
            "__MATERIALS__", esc(site["materials"])
        )
        .replace("__ERA__", esc(site["era"]) or "Heritage Site")
        .replace("__FEATURES__", esc(site["features"])),
        unsafe_allow_html=True,
    )

    # --- VISUAL GALLERY ----------------------------------------------------
    gallery = site["gallery"]
    if gallery:
        st.markdown(
            "<div class='site-section-title'>In Pictures</div>",
            unsafe_allow_html=True,
        )
        rows = [gallery[i : i + 3] for i in range(0, len(gallery), 3)]
        for row in rows:
            cells = []
            for img in row:
                img_url = img["image_url"]
                cap = img.get("title", "") or "Wikimedia Commons"
                cap_name = esc(cap)
                if len(cap_name) > 48:
                    cap_name = cap_name[:48].rsplit(" ", 1)[0] + "…"
                cells.append(
                    f"<div class='gallery-cell'>"
                    f"<img class='gallery-img' src='{esc(img_url, quote=True)}' alt='{cap_name}'>"
                    f"<div class='gallery-cap'>{cap_name}</div>"
                    f"</div>"
                )
            st.markdown(
                f"<div class='gallery-row'>{''.join(cells)}</div>",
                unsafe_allow_html=True,
            )

    # --- WHAT TO LOOK FOR -------------------------------------------------
    if site["highlights"]:
        st.markdown(
            "<div class='site-section-title'>What to Look For</div>",
            unsafe_allow_html=True,
        )
        lis = "".join(
            f"<li>{esc(h)}</li>" for h in site["highlights"]
        )
        st.markdown(
            f"<ul class='highlight-list'>{lis}</ul>",
            unsafe_allow_html=True,
        )

    # --- INTERACTIVE AI ASSISTANT CARD -------------------------------------
    st.markdown(
        f"""
        <div class='ai-card'>
            <div class='ai-card-title'>Ask about {name}</div>
            <div class='ai-card-sub'>Chat with the archive curator about this building's history and architecture.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestion_questions = [
        f"How was {site['name']} constructed?",
        f"What is the historical significance of {site['name']}?",
        f"What materials were used to build {site['name']}?",
    ]
    with st.container(key=f"ai_pills_{site_id}"):
        pill_cols = st.columns(3)
        for i, (col, sq) in enumerate(zip(pill_cols, suggestion_questions)):
            if col.button(sq, key=f"sug_{site_id}_{i}", width="stretch"):
                ans, srcs = query_rag_safe(sq, site_id=site_id)
                _pill_ask(site, sq, ans, srcs)

    # Chat history for this site
    if chat:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        for msg in chat:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-bubble'>{esc(msg['content'])}</div><div style='clear: both;'></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='assistant-bubble'>{_assistant_html(msg.get('content', ''), msg.get('sources'))}</div><div style='clear: both;'></div>",
                    unsafe_allow_html=True,
                )

    # Inline search/chat input with leading icon + circular action button
    ask_col, send_col = st.columns([6, 1])
    with st.container(key=f"askbar_{site_id}"):
        with ask_col:
            site_q = st.text_input(
                "Ask a question",
                key=f"site_q_input_{site_id}",
                placeholder="e.g. Why was it built beside the creek?",
                label_visibility="collapsed",
            )
        with send_col:
            sent = st.button("➤", key=f"site_q_btn_{site_id}", width="stretch")
    if sent and site_q.strip():
        ans, srcs = query_rag_safe(site_q, site_id=site_id)
        _pill_ask(site, site_q, ans, srcs)


# Modal Trigger Execution
if st.session_state.active_site_id:
    site_data = get_site_by_id(st.session_state.active_site_id)
    if site_data:
        render_site_dialog(site_data)
        st.session_state.active_site_id = None


# ==============================================================================
# VIEW 1: LANDING SCREEN
# ==============================================================================
if st.session_state.view == "landing":
    st.markdown(
        "<style>"
        "section[data-testid='stSidebar']{display:none!important;}"
        "header[data-testid='stHeader'] [data-testid='stExpandSidebarButton']{display:none!important;}"
        "header[data-testid='stHeader'] [data-testid='stCollapseSidebarButton']{display:none!important;}"
        ".block-container{padding-top:2.5rem;}"
        "</style>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='landing-body'>"
        "<div class='landing-tag'>A Digital Archive of Dubai</div>"
        "<h1 class='landing-title'>Athar.<br>Every place leaves a trace.</h1>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(key="landing_cta"):
        if st.button("Let's explore!", key="landing_go"):
            st.session_state.view = "main"
            st.rerun()

    st.markdown(
        "<p class='landing-note'>An independent digital heritage archive · "
        "curated from Wikipedia, Wikimedia Commons &amp; Dubai Culture</p>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# VIEW 2: MAIN ARCHIVE MAP & CATALOG
# ==============================================================================
elif st.session_state.view == "main":

    render_sidebar_chat_library()

    with st.container(key="back_link"):
        if st.button("← Back to Welcome"):
            st.session_state.view = "landing"
            st.rerun()

    st.markdown(
        "<div class='main-header'>"
        "<p class='main-header-tag'>A Digital Archive of Old Dubai</p>"
        "<h1 class='main-header-title'>Athar<span class='main-title-dot'>.</span></h1>"
        "<p class='main-header-sub'>Wander the creek, tap a pin, and open the record of a fort, "
        "a wind-tower house or a souk that built this city — then ask the curator anything about it.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # MAP RENDERING
    # --------------------------------------------------------------------------
    st.markdown(
        "### Interactive Creek Map",
        help="Click a pin on the map to open its archive record",
    )

    if _map_library_available and SITES:
        creek_map = folium.Map(
            location=[25.2048, 55.2708],
            zoom_start=12,
            control_scale=True,
            width="100%",
            height="100%",
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            name="Esri World Street Map",
        ).add_to(creek_map)

        site_markers = folium.FeatureGroup(name="Sites")
        for site in SITES:
            popup_html = (
                "<div style='font-family: Georgia, serif; min-width: 190px;'>"
                f"<div style='font-weight: 700; font-size: 14px; color: #4A3B2F;'>{html.escape(site['name'])}</div>"
                f"<div style='color: #8C7B70; font-size: 12px; margin-top: 2px;'>{html.escape(site['district'])}</div>"
                f"<div style='color: #6B5B4E; font-size: 12px; margin-top: 6px;'>{html.escape(site['blurb'])}</div>"
                "</div>"
            )
            _tt = (
                f"<div style='font-family: Georgia, serif; min-width: 200px; max-width: 280px; "
                f"word-wrap: break-word; overflow-wrap: break-word; white-space: normal;'>"
                f"<b style='font-size: 13px; color: #4A3B2F;'>{html.escape(site['name'])}</b><br>"
                f"<span style='color: #8C7B70; font-size: 11px;'>{html.escape(site['district'])}</span><br>"
                f"<span style='color: #6B5B4E; font-size: 11px; line-height: 1.3; "
                f"word-wrap: break-word; overflow-wrap: break-word;'>{html.escape(site['blurb'])}</span>"
                f"</div>"
            )
            folium.Marker(
                location=[site["lat"], site["lon"]],
                tooltip=folium.Tooltip(_tt, sticky=True),
                popup=folium.Popup(popup_html, max_width=280),
                icon=folium.DivIcon(
                    html=(
                        "<div style='font-size: 22px; line-height: 26px; "
                        "text-shadow: 0 1px 2px rgba(0,0,0,.45);'>📍</div>"
                    ),
                    icon_size=(26, 26),
                    icon_anchor=(13, 24),
                ),
            ).add_to(site_markers)
        site_markers.add_to(creek_map)

        clicked = st_folium(
            creek_map,
            width="stretch",
            height=560,
            feature_group_to_add=[site_markers],
        )

        marker_click = clicked.get("last_object_clicked")
        if marker_click is not None:
            click_lat = marker_click.get("lat")
            click_lng = marker_click.get("lng")
            if click_lat is not None and click_lng is not None:
                click_key = (click_lat, click_lng)
                if click_key != st.session_state.get("_handled_pin"):
                    st.session_state._handled_pin = click_key
                    closest_site = None
                    closest_dist = float("inf")
                    for s in SITES:
                        d = ((s["lat"] - click_lat) ** 2 + (s["lon"] - click_lng) ** 2) ** 0.5
                        if d < closest_dist:
                            closest_dist = d
                            closest_site = s
                    if (
                        closest_site
                        and closest_dist < 0.004
                        and closest_site["id"] != st.session_state.get("active_site_id")
                    ):
                        st.session_state.active_site_id = closest_site["id"]
                        st.rerun()
    else:
        st.warning(
            f"Interactive map library not installed in this Python environment ({_map_error}). "
            "Run with: .venv/bin/python -m streamlit run app2.py"
        )

    # --------------------------------------------------------------------------
    # GENERAL RAG CHAT + PHOTO SEARCH
    # --------------------------------------------------------------------------
    with st.container(key="chat_band"):
        with st.container(key="chat_head"):
            hcol_l, hcol_c, hcol_r = st.columns([1, 2, 1], vertical_alignment="center")
            with hcol_c:
                st.markdown(
                    "<h3 class='chat-band-title'>Ask the archive anything</h3>",
                    unsafe_allow_html=True,
                )
            with hcol_r:
                if st.button("➕ New chat", key="new_chat", help="Start a fresh chat with a new question"):
                    if st.session_state.chat_history:
                        save_general_chat()
                    st.session_state.chat_history = []
                    st.session_state.general_conv_stem = None
                    st.session_state.last_archive_question = ""
                    st.rerun()
        st.markdown(
            "<p class='chat-band-sub'>Wind towers, pearl diving, coral-stone walls — "
            "the curator answers questions about old Dubai in general.</p>",
            unsafe_allow_html=True,
        )

        _render_general_messages(st.session_state.chat_history)

        st.markdown(
            "<div class='site-kicker-label' style='margin-top: 1.3rem;'>Try asking</div>",
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(3)
        for chip_col, text, key in zip(
            chip_cols,
            [
                "What is a barjeel?",
                "Which site is the oldest?",
                "Plan me a half-day heritage walk",
            ],
            ["chip_0", "chip_1", "chip_2"],
        ):
            with chip_col:
                if st.button(text, key=key):
                    answer_general_question(text)

        with st.container(key="photo_bar"):
            ph_col, id_col = st.columns([4, 1])
            with ph_col:
                photo_upload = st.file_uploader(
                    "Search by photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="photo_upload",
                    label_visibility="collapsed",
                )
            with id_col:
                identify = st.button("Identify", key="photo_go", width="stretch")
        if identify and photo_upload is not None:
            _run_photo_search(photo_upload)

        with st.container(key="chat_askbar"):
            gen_col, ask_col = st.columns([5, 1])
            with gen_col:
                gen_q = st.text_input(
                    "Enter your query:",
                    placeholder="e.g. Why did Dubai's houses have wind towers?",
                    label_visibility="collapsed",
                )
            with ask_col:
                consult = st.button("Ask", width="stretch")
        if consult and gen_q.strip():
            answer_general_question(gen_q)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # CARDS GRID
    # --------------------------------------------------------------------------
    st.markdown(
        "<h3 style='font-family: Fraunces, Georgia, serif;'>In the archive</h3>",
        unsafe_allow_html=True,
    )

    for i in range(0, len(SITES), 3):
        grid_cols = st.columns(3)
        for idx, site in enumerate(SITES[i : i + 3]):
            with grid_cols[idx]:
                name = html.escape(site["name"])
                district = html.escape(site["district"])
                built = html.escape(site.get("builtDate") or "")

                if site["image"]:
                    square = f"<img class='site-card-img' src='{html.escape(site['image'], quote=True)}' alt='{name}'>"
                else:
                    square = f"<div class='site-square-empty'>{name}</div>"

                meta = f"{district} <b>· {built}</b>" if built else district

                st.markdown(
                    "<div class='site-card'>"
                    f"{square}"
                    f"<p class='site-card-meta'>{meta}</p>"
                    f"<p class='site-card-name'>{name}</p>"
                    f"<p class='site-card-blurb'>{html.escape(site['blurb'])}</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Open the archive record →",
                    key=f"card_btn_{site['id']}",
                    width="stretch",
                ):
                    st.session_state.active_site_id = site["id"]
                    st.rerun()

    st.markdown(
        "<p class='archive-footer'>Athar — an independent digital heritage archive. "
        "Descriptions are curated summaries.</p>",
        unsafe_allow_html=True,
    )