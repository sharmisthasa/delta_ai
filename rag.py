"""The four steps of RAG, one function each: chop, convert, store, retrieve.

Read this file top-to-bottom and you'll understand how Retrieval-Augmented Generation
actually works. No frameworks, no hidden magic.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
import streamlit as st

import chromadb
import requests
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "reviewers"
CHAT_MODEL = "openai/gpt-oss-120b"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "SikkaDubaiHeritageArchive/1.0 (student project)"}
DATA_FILE = Path(__file__).parent / "dubai_buildings_full.json"
NO_ANSWER_MARKER = "NO_ANSWER"
MAX_PAGE_CHARS = 3000

# Global lazy Groq client
_groq_client = None


def get_groq_client() -> Groq:
    """Lazy initialize Groq client so importing rag.py doesn't crash if the key isn't loaded."""
    global _groq_client
    if _groq_client is None:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is missing. "
                "Ensure your .env file contains GROQ_API_KEY=your_key."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# Chroma's default local embedding model converts text into meaning-based numbers.
_embedding_model = DefaultEmbeddingFunction()

# --- Step 0: one-time setup (ChromaDB lives on disk so your data survives restarts) ---

_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    return _client.get_or_create_collection(name=COLLECTION_NAME)


# --- Step 1: CHOP — split long text into small pieces ---

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split a big string into overlapping pieces of ~`size` characters each.

    Why chunks? Because AI models work better with small focused passages than
    with huge walls of text. Why overlap? So a sentence that straddles a chunk
    boundary still appears fully in at least one chunk.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        chunks.append(text[start : start + size])
        start += step

    return chunks


# --- Step 2: CONVERT — turn each chunk into a list of numbers (an "embedding") ---

def embed_text(text: str) -> list[float]:
    """Use Chroma's local embedding model to turn text into meaning-based numbers.

    Two similar-meaning sentences become two similar lists of numbers.
    That's the whole trick.
    """
    embedding = _embedding_model([text])[0]

    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return list(embedding)


# --- Step 3: STORE — put all the chunks + their embeddings into ChromaDB ---

def embed_and_store(chunks: list[str], filename: str) -> int:
    """Embed every chunk of a file and save them. Returns how many chunks were stored."""
    collection = get_collection()

    # If this file was uploaded before, remove the old chunks first.
    try:
        collection.delete(where={"filename": filename})
    except Exception:
        pass

    if not chunks:
        return 0

    ids = [
        f"{filename}__chunk_{i}"
        for i in range(len(chunks))
    ]

    embeddings = [
        embed_text(chunk)
        for chunk in chunks
    ]

    metadatas = [
        {
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


# --- Step 4: RETRIEVE — given a question, find the most relevant chunks ---

def retrieve(question: str, k: int = 3) -> list[dict]:
    """Return the top-k chunks most relevant to the question, with their source filenames."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    question_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
    )

    # ChromaDB returns results as lists-of-lists (one per query). We only sent one.
    hits = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        hits.append(
            {
                "text": document,
                "filename": metadata["filename"],
            }
        )

    return hits


# --- Step 5: ANSWER — combine retrieved chunks + question, send to Groq ---

# Answers are pitched at curious teenagers and young adults with short
# attention spans: snappy, scannable, and broken into bullet points rather
# than long walls of text.
RAG_PROMPT = """You are Sikka, a friendly guide to Dubai's heritage architecture, chatting with a curious teenager or young adult.

Rules:
- Sound upbeat, warm and casual, like a friend sharing a cool fact — but still factual.
- Keep it snappy and scannable: lead with a short intro line, then break the main points into bullet points ("• ...").
- Use short sentences and everyday words. No jargon dumps.
- Ground every claim in the excerpts below. If a detail is not in the excerpts, don't invent it.
- Do not cite sources inside your reply — no "Excerpt", "Source", or page names. The app shows the source links below your answer.
- If the excerpts do not contain enough information to answer the question, reply with exactly the single word {marker} by itself.

Excerpts:
{context}

User's question: {question}

Your answer:"""

WIKIPEDIA_PROMPT = """You are Sikka, a friendly guide to Dubai's heritage architecture, chatting with a curious teenager or young adult.

The archive could not answer the question, so the Wikipedia text below has been consulted instead.

Rules:
- Sound upbeat, warm and casual, like a friend sharing a cool fact — but still factual.
- Keep it snappy and scannable: lead with a short intro line, then break the main points into bullet points ("• ...").
- Use short sentences and everyday words. No jargon dumps.
- Ground every claim in the Wikipedia text below. If a detail is not there, don't invent it.
- Do not cite the Wikipedia page inside your reply. The app shows the source links below your answer.
- If the text still does not answer the question, reply with exactly the single word {marker} by itself.

Wikipedia text:
{context}

User's question: {question}

Your answer:"""

NOT_AVAILABLE_PROMPT = """You are Sikka, a friendly guide to Dubai's heritage architecture, chatting with a curious teenager or young adult.

You could not find reliable information to answer the visitor's question anywhere in the heritage archive or on Wikipedia.

Rules:
- Keep your whole reply to just 1-2 short, casual sentences. Do not ramble and do not make up facts.
- Be honest: say you couldn't find reliable info on that.
- Then offer 1-2 alternative topics the archive DOES cover (for example: wind towers / barjeel, coral & gypsum houses, pearling, souks, or one of Dubai's historical buildings) — phrased as a friendly "want to know about X instead?"

User's question: {question}

Your answer:"""


def _generate(prompt: str) -> str:
    """Send one prompt to the chat model and return the raw text it produces."""
    completion = get_groq_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


# Curated notes from the Dubai Culture & Arts Authority site (dubaiculture.gov.ae),
# keyed by the slug used for each site record. Consulted before live Wikipedia so
# answers reflect the authority's official descriptions of architecture + history.
DCULTURE_NOTES = {
    "al-fahidi-historical-neighbourhood": {
        "text": (
            "Nestled along Dubai Creek, Al Fahidi Historical Neighbourhood reflects the "
            "traditional lifestyle of Dubai from the mid-19th century to the 1970s. Its "
            "buildings with high air towers (barajeel) were built with traditional "
            "materials such as stone, gypsum, teak, sandalwood, fronds and palm wood, "
            "aligned side by side and separated by alleys, pathways and public squares. "
            "Owing to its strategic location at Dubai Creek, the district played an "
            "important role in managing Dubai and organising its commercial relations "
            "overseas. The neighbourhood now hosts cultural and artistic activities: "
            "art galleries, specialised museums, cultural centres such as the Sheikh "
            "Mohammed Center for Cultural Understanding, and seasonal events like the "
            "Sikka Art Festival."
        ),
        "label": "Dubai Culture — Al Fahidi Historical Neighbourhood",
        "url": "https://dubaiculture.gov.ae/en/attractions/heritage-sites/al-fahidi-historical-neighbourhood",
    },
    "heritage-house": {
        "text": (
            "Located in one of the oldest urban sites in Dubai, Heritage House was "
            "established by Mr. Matar Saeed bin Mazina in 1890 to describe the nature "
            "of traditional local housing. Visitors see household tools and utensils of "
            "pottery, copper, wood and glass; furniture, clothes, jewellery, cosmetics "
            "and historical toys of both local and imported make, from India and the "
            "East coast of Africa. The house shows the elements of historical heritage "
            "homes — rooms, diwans (grand halls) and air towers (barajeel) — and how "
            "their design and distribution revolve around the inner courtyard, around "
            "which all the house components were arranged."
        ),
        "label": "Dubai Culture — Heritage House",
        "url": "https://dubaiculture.gov.ae/en/attractions/heritage-sites/heritage-house",
    },
    "al-ahmadiya-school": {
        "text": (
            "Al Ahmadiya School, opened in 1912 and founded by the pearl merchant "
            "Sheikh Ahmed Bin Dalmouk, was Dubai's first formal school. It began with "
            "around thirty pupils studying Arabic, mathematics and Islamic studies. "
            "The coral and gypsum building is arranged around a central courtyard with "
            "classrooms and carved teak doors on all sides. It was restored and "
            "reopened as a museum showing how education in Dubai evolved from informal "
            "Quranic study to a modern system."
        ),
        "label": "Dubai Culture — Al Ahmadiya School",
        "url": "https://dubaiculture.gov.ae/en/attractions/heritage-sites/al-ahmadiya-school",
    },
    "majlis-ghorfat-umm-al-sheif": {
        "text": (
            "Despite its small size, Majlis Ghorfat Umm Al Sheif is a special historical "
            "monument distinguished by its traditional heritage. It was established in "
            "1955 as the summer retreat of the late Sheikh Rashid bin Saeed in the "
            "Jumeirah area, famous for its calm, tranquil atmosphere and palm trees. The "
            "single-storey house is built from adobe, gypsum and coral with timber, and "
            "its roof terrace was used to dry dates and as a cool open-air sleeping "
            "platform. It gives visitors a chance to see the historical furniture and "
            "tools of a traditional Emirati majlis — copper buckets, coffee stoves, "
            "earthenware, brassware, rugs and carpets, along with clocks and radios and "
            "men's rifles and daggers. The majlis was fully restored in 1994 with a "
            "small garden of date palms and a traditional aflaj irrigation system."
        ),
        "label": "Dubai Culture — Majlis Ghorfat Umm Al Sheif",
        "url": "https://dubaiculture.gov.ae/en/attractions/heritage-sites/majlis-ghorfat-umm-al-sheif",
    },
    "al-fahidi-fort": {
        "text": (
            "Al Fahidi Fort is Dubai's oldest standing building, raised around 1787 to "
            "guard the landward approaches to the settlement on the creek, and since "
            "1971 it has served as the Dubai Museum. Its watchtower and thick coral and "
            "gypsum walls kept watch over the city, and its courtyard opens onto "
            "underground galleries with dioramas of pearling, souk life and the desert "
            "Bedouin, together with artefacts tracing the emirate back over 3,000 years."
        ),
        "label": "Dubai Museum & Al Fahidi Fort",
        "url": "https://dubaiculture.gov.ae/en",
    },
    "al-shindagha-heritage-district": {
        "text": (
            "Al Shindagha is the historic creekside district where Dubai's ruling "
            "family and its merchants built coral-stone courtyard houses from the "
            "mid-19th century. It is now the setting of the Al Shindagha Museum — "
            "Dubai's largest heritage museum — made up of restored houses along the "
            "creek that tell the story of the emirate, from its founding to the "
            "present, with themed houses covering governance, poetry, jewellery and "
            "traditional medicine."
        ),
        "label": "Al Shindagha Museum",
        "url": "https://alshindagha.dubaiculture.gov.ae/en",
    },
    "sheikh-saeed-al-maktoum-house": {
        "text": (
            "Built in 1896 as the seat of the Al Maktoum family and home of Sheikh "
            "Saeed bin Maktoum, who ruled Dubai from 1912 to 1958, this long, low "
            "creekside residence is part of the wider Al Shindagha Museum complex. Its "
            "two wind towers kept the interiors cool and its rooms open onto a "
            "courtyard facing the water. As a museum its nine wings display photographs "
            "of old Dubai, maps, marine life, coins and stamps, and the family history "
            "that shaped the city."
        ),
        "label": "Dubai Culture — Sheikh Saeed Al Maktoum House",
        "url": "https://dubaiculture.gov.ae/en",
    },
}


def _dataset_wiki_url(site_name: str) -> str | None:
    """Find the curated Wikipedia URL for a site name stored in the dataset."""
    if not site_name or not DATA_FILE.exists():
        return None
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, ValueError):
        return None
    for record in records:
        if _slugify(record.get("site_name", "")) != _slugify(site_name):
            continue
        return (record.get("wikipedia") or {}).get("wikipedia_url") or None
    return None


def _wiki_page_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"


def _attach_links(hits: list[dict]) -> list[dict]:
    """Annotate each retrieved chunk with a clickable source url + label."""
    linked = []
    for hit in hits:
        url = hit.get("url") or _dataset_wiki_url(hit["filename"]) or _wiki_page_url(
            hit["filename"]
        )
        linked.append(
            {
                **hit,
                "label": hit.get("label") or hit["filename"],
                "url": url,
            }
        )
    return linked


def _local_site_wikipedia(site_id: str) -> dict | None:
    """Return the curated Wikipedia page text stored in the dataset for a site."""
    if not site_id or not DATA_FILE.exists():
        return None

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, ValueError):
        return None

    for record in records:
        if _slugify(record.get("site_name", "")) != _slugify(site_id):
            continue
        full_text = (record.get("wikipedia") or {}).get("full_text", "")
        if full_text:
            return {
                "text": full_text[:MAX_PAGE_CHARS],
                "filename": record["site_name"],
            }
        return None
    return None


def _wikipedia_api_get(params: dict) -> requests.Response:
    """GET the Wikipedia API, backing off politely when rate limited."""
    for attempt in range(3):
        time.sleep(0.5)
        response = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers=WIKI_HEADERS,
            timeout=20,
        )
        if response.status_code == 429 and attempt < 2:
            time.sleep(float(response.headers.get("Retry-After", "3")))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("Wikipedia API rate limit exceeded.")


def _search_wikipedia(question: str, limit: int = 2) -> list[dict]:
    """Search Wikipedia live and pull short text from the top matching pages."""
    search = _wikipedia_api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": f"{question} Dubai",
            "srnamespace": 0,
            "srlimit": limit,
            "format": "json",
        }
    ).json()

    pages = []
    for match in search.get("query", {}).get("search", []):
        extracts = _wikipedia_api_get(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": match["title"],
                "format": "json",
            }
        ).json()
        for page in extracts.get("query", {}).get("pages", {}).values():
            text = (page.get("extract") or "").strip()[:MAX_PAGE_CHARS]
            if not text:
                continue
            pages.append(
                {"text": text, "filename": page.get("title", match["title"])}
            )
    return pages


def _wikipedia_fallback(question: str, site_id: str) -> list[dict]:
    """Gather background text to consult when the archive can't answer.

    Prefers the curated Dubai Culture notes and the page already stored for the
    site being discussed, then falls back to a live Wikipedia search.
    """
    if site_id and site_id in DCULTURE_NOTES:
        note = DCULTURE_NOTES[site_id]
        return [
            {
                "text": note["text"],
                "filename": note["label"],
                "label": note["label"],
                "url": note["url"],
            }
        ]
    local = _local_site_wikipedia(site_id)
    if local:
        return [local]
    return _search_wikipedia(question)


def answer_with_sources(
    question: str,
    k: int = 3,
    site_id: str = None,
) -> tuple[str, list[dict], str]:
    """Full RAG pipeline: retrieve, build prompt, ask Groq, return (answer, sources, status).

    Answers in a warm, teen-friendly voice with bullet points. The local archive
    always wins, then Wikipedia, and only if neither can answer does it fall
    back to a short "not available — want to know about X instead?" reply.
    Status is "archive", "wikipedia", or "notavailable" so the UI can reliably
    trigger a web search when the archive and Wikipedia both come up short.
    """
    hits = retrieve(question, k=k)

    if hits:
        context = "\n\n".join(
            f"Source {i + 1} ({hit['filename']}):\n{hit['text']}"
            for i, hit in enumerate(hits)
        )
        answer = _generate(
            RAG_PROMPT.format(
                context=context,
                question=question,
                marker=NO_ANSWER_MARKER,
            )
        )
        if not answer.strip().upper().startswith(NO_ANSWER_MARKER):
            return answer, _attach_links(hits), "archive"

    wiki_hits = _wikipedia_fallback(question, site_id)
    if wiki_hits:
        context = "\n\n".join(
            f"Source {i + 1} ({hit['filename']}):\n{hit['text']}"
            for i, hit in enumerate(wiki_hits)
        )
        answer = _generate(
            WIKIPEDIA_PROMPT.format(
                context=context,
                question=question,
                filename=wiki_hits[0]["filename"],
                marker=NO_ANSWER_MARKER,
            )
        )
        if not answer.strip().upper().startswith(NO_ANSWER_MARKER):
            return answer, _attach_links(wiki_hits), "wikipedia"

    # Nothing usable anywhere: give a short, honest "not available" reply that
    # points the visitor at topics the archive does cover.
    answer = _generate(
        NOT_AVAILABLE_PROMPT.format(question=question)
    )
    return answer, [], "notavailable"


# --- Helpers for the UI ---

def list_ingested_filenames() -> list[str]:
    collection = get_collection()

    if collection.count() == 0:
        return []

    metadatas = collection.get()["metadatas"]

    return sorted(
        {
            metadata["filename"]
            for metadata in metadatas
        }
    )


def clear_all():
    """Delete every chunk. Used by the reset button in the UI."""
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def read_pdf(file) -> str:
    reader = PdfReader(file)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n\n".join(pages)


def read_txt(file) -> str:
    raw = file.read()

    if isinstance(raw, bytes):
        return raw.decode(
            "utf-8",
            errors="ignore",
        )

    return raw


def extract_text_from_upload(uploaded_file) -> str:
    """Dispatch on extension and return the raw text of a .pdf or .txt upload."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)

    if name.endswith(".txt"):
        return read_txt(uploaded_file)

    raise ValueError(
        f"Unsupported file type: {uploaded_file.name}"
    )


MAX_FILE_MB = 2
MAX_FILES = 5


def file_too_big(uploaded_file) -> bool:
    size_mb = uploaded_file.size / (1024 * 1024)
    return size_mb > MAX_FILE_MB