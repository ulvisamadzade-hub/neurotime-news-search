import os
import pickle
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")

_metadata: list[dict] = []

AZ_STOPWORDS = {
    # Conjunctions / particles
    "və", "bu", "bir", "ilə", "üçün", "olan", "ki", "da", "də", "o", "onun",
    "həm", "lakin", "isə", "daha", "sonra", "artıq", "hər", "çox", "ən",
    "belə", "buna", "bunun", "ancaq", "amma", "yalnız", "bütün", "digər",
    "həmçinin", "haqqında", "barədə", "üzrə", "kimi", "qədər", "öz",
    "həmin", "hansı", "nə", "necə", "nəyi", "hansısa", "bunlar", "onlar",
    "bizim", "sizin", "onların", "özü", "özünü", "biz", "siz", "yox", "var",
    "əsasən", "olaraq", "üzərində", "arasında", "olmaq",
    # Generic verbs and conjugations
    "edir", "edilib", "olub", "olacaq", "edib", "verib", "bildirib",
    "keçirildi", "bildirilib", "etdi", "dedi", "görə", "edilməsi", "olması",
    "keçirilməsi", "aparılması", "edilən", "olunur", "edilir", "bildirir",
    "verir", "alır", "olur", "gedir", "gəlir", "görür", "bilir", "deyir",
    "istər", "baxır", "saxlayır", "aparır", "keçir", "çıxır", "tutulub",
    "aparılıb", "verilib", "olunub", "edilmişdir", "olunmuşdur", "bildirilir",
    "qeyd", "etmək", "olduğunu", "etdiyini", "verdiyini", "aldığını",
    "getdiyini", "etməsi", "verməsi", "alması", "başlaması", "açıqladı",
    "bildirdi", "qeyd etdi", "açıqladı", "açıqlayıb", "vurğuladı",
    "xatırlatdı", "əlavə etdi", "nəzərə çatdırdı", "söylədi",
    # Common adjectives / adverbs that are too generic
    "baş", "bağlı", "əsas", "xüsusi", "milli", "böyük", "kiçik", "yeni",
    "köhnə", "tam", "lazım", "əlaqədar", "əlaqəli", "müvafiq", "müəyyən",
    "cari", "keçən", "gələn", "növbəti", "əvvəlki", "son", "ilk", "hazır",
    "rəsmi", "ictimai", "ümumi", "xarici", "daxili", "yerli", "beynəlxalq",
    # Common generic nouns (too broad to be keywords)
    "il", "ay", "gün", "vaxt", "dövr", "müddət", "nəfər", "şəxs", "yer",
    "məkan", "hal", "durum", "vəziyyət", "sistem", "sahə", "məsələ", "sənəd",
    "qərar", "nəticə", "proses", "şərait", "imkan", "iclас", "iclas",
    "toplantı", "görüş", "müzakirə", "razılıq", "müqavilə", "sərəncam",
    "xəbər", "məlumat", "hesabat", "açıqlama", "bəyanat", "müraciət",
    "proqram", "plan", "layihə", "tədbirlər", "tədbird", "addım", "çərçivə",
    "istiqamət", "mərhələ", "tədbir",
    # English stopwords
    "the", "and", "for", "was", "are", "with", "that", "this", "from",
    "have", "has", "been", "will", "its", "not", "but", "they", "said",
    "also", "after", "about", "which", "their", "there", "would",
}

MIN_WORD_LEN  = 4
MIN_ENTITY_FREQ = 2
MIN_WORD_FREQ   = 8   # raised — proper nouns still dominate at high frequency


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

    word_counts:   Counter = Counter()
    entity_counts: Counter = Counter()

    # Multi-word proper entities: two+ Title-case words
    entity_pattern = re.compile(
        r"\b[A-ZƏÜÖĞŞÇİ][a-zəüöğşçı]+(?:[-\s][A-ZƏÜÖĞŞÇİ][a-zəüöğşçı]+)+\b"
    )
    # Single words (any case) — we filter to capitalized below
    word_pattern = re.compile(r"\b[A-ZƏÜÖĞŞÇİa-zəüöğşçı]{4,}\b")

    for text in texts:
        if not text:
            continue

        # Multi-word entities first (e.g. "Mərkəzi Bank", "Kapital Bank")
        for entity in entity_pattern.findall(text):
            if len(entity) > 4:
                entity_counts[entity] += 1

        # Single words — ONLY count when capitalized in source text (proper nouns)
        for word in word_pattern.findall(text):
            if not word[0].isupper():
                continue  # skip lowercase — filters out verbs, adjectives, etc.
            lower = word.lower()
            if lower not in AZ_STOPWORDS and len(lower) >= MIN_WORD_LEN:
                word_counts[lower] += 1

    combined: Counter = Counter()

    for entity, cnt in entity_counts.items():
        if cnt >= MIN_ENTITY_FREQ:
            combined[entity] = cnt

    for word, cnt in word_counts.items():
        if word not in combined and cnt >= MIN_WORD_FREQ:
            combined[word] = cnt

    return combined.most_common(top_n)
