import os
import pickle
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")

_metadata: list[dict] = []

# Common Azerbaijani stopwords + noise
AZ_STOPWORDS = {
    "və", "bu", "bir", "ilə", "üçün", "olan", "ki", "da", "də", "o", "onun",
    "həm", "lakin", "isə", "əsasən", "daha", "sonra", "artıq", "hər", "çox",
    "ən", "belə", "buna", "bunun", "edir", "edilib", "olub", "olacaq", "var",
    "yox", "biz", "siz", "onlar", "həmçinin", "haqqında", "barədə", "üzrə",
    "kimi", "qədər", "ancaq", "amma", "yalnız", "bütün", "digər", "yeni",
    "keçirildi", "bildirilib", "qeyd", "etdi", "dedi", "görə", "öz",
    "edilməsi", "olması", "keçirilməsi", "aparılması", "olaraq", "edib",
    "verib", "bildirib", "təşkil", "məlumat", "xəbər", "bildirir", "etmək",
    "üzərində", "arasında", "olan", "olmaq", "edilən", "olunur", "edilir",
    "həmin", "hər", "hansı", "nə", "necə", "nəyi", "hansısa", "belə",
    "bunlar", "onlar", "bizim", "sizin", "onların", "özü", "özünü",
    "olduğunu", "etdiyini", "verdiyini", "aldığını", "getdiyini",
    "the", "and", "for", "was", "are", "with", "that", "this", "from",
    "have", "has", "been", "will", "its", "not", "but", "they",
}

# Minimum chars and frequency thresholds
MIN_WORD_LEN = 3
MIN_ENTITY_FREQ = 2
MIN_WORD_FREQ = 5


def load_metadata():
    global _metadata
    with open(META_PATH, "rb") as f:
        _metadata = pickle.load(f)


def _iter_texts(articles: list[str] | None) -> list[str]:
    if articles:
        return articles
    if not _metadata:
        load_metadata()
    return [f"{m.get('title', '')} {m.get('snippet', '')}" for m in _metadata]


def extract_keywords(articles: list[str] | None = None, top_n: int = 20) -> list[tuple[str, int]]:
    texts = _iter_texts(articles)

    word_counts: Counter = Counter()
    entity_counts: Counter = Counter()

    # Regex: Azerbaijani and Latin alphabet, handles special chars ə ü ö ğ ş ç ı İ Ə etc.
    word_pattern = re.compile(r"\b[A-ZƏÜÖĞŞÇIa-zəüöğşçıİ]{3,}\b")
    # Multi-word entities: two or more Title-case words in a row
    entity_pattern = re.compile(
        r"\b[A-ZƏÜÖĞŞÇİ][a-zəüöğşçı]+(?:[-\s][A-ZƏÜÖĞŞÇİ][a-zəüöğşçı]+)+\b"
    )

    for text in texts:
        if not text:
            continue

        # multi-word entities first
        for entity in entity_pattern.findall(text):
            if len(entity) > 4:
                entity_counts[entity] += 1

        # single words
        for word in word_pattern.findall(text):
            lower = word.lower()
            if lower not in AZ_STOPWORDS and len(lower) >= MIN_WORD_LEN and not lower.isdigit():
                word_counts[lower] += 1

    # Merge: entities take priority, single words fill the rest
    combined: Counter = Counter()
    for entity, cnt in entity_counts.items():
        if cnt >= MIN_ENTITY_FREQ:
            combined[entity] = cnt

    for word, cnt in word_counts.items():
        # skip if already captured as part of a multi-word entity
        if word not in combined and cnt >= MIN_WORD_FREQ:
            combined[word] = cnt

    return combined.most_common(top_n)
