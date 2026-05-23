BANKS = [
    {
        "canonical": "ABB",
        "aliases": [
            "ABB", "International Bank of Azerbaijan", "Beynəlxalq Bank",
            "Azerbaycan Beynəlxalq Bank", "abb-bank",
        ],
    },
    {
        "canonical": "Kapital Bank",
        "aliases": [
            "Kapital Bank", "KapitalBank", "Kapital Bankı", "Kapital Bankın",
            "kapitalbank",
        ],
    },
    {
        "canonical": "PASHA Bank",
        "aliases": [
            "PASHA Bank", "PASHAbank", "PAŞA Bank", "PASHA Bankı",
            "PASHA Bankın", "pashabank",
        ],
    },
    {
        "canonical": "Xalq Bank",
        "aliases": [
            "Xalq Bank", "XalqBank", "Xalq Bankı", "Xalq Bankın", "xalqbank",
        ],
    },
    {
        "canonical": "AccessBank",
        "aliases": [
            "AccessBank", "Access Bank", "AccessBankın", "AccessBankı",
            "accessbank",
        ],
    },
    {
        "canonical": "AFB Bank",
        "aliases": [
            "AFB Bank", "AFB", "AFB Bankı", "AFB Bankın", "afb.az",
        ],
    },
    {
        "canonical": "Bank of Baku",
        "aliases": [
            "Bank of Baku", "BankofBaku", "Bank of Bakuya", "bankofbaku",
        ],
    },
    {
        "canonical": "Bank Respublika",
        "aliases": [
            "Bank Respublika", "Bankrespublika", "Bank Respublikanın",
            "Bank Respublikası", "bankrespublika",
        ],
    },
    {
        "canonical": "Bank BTB",
        "aliases": [
            "Bank BTB", "BTB Bank", "BTB", "btb.az",
        ],
    },
    {
        "canonical": "Bank Eurasia",
        "aliases": [
            "Bank Eurasia", "BankAvrasiya", "Bank Avrasiya", "Avrasiya Bankı",
            "bankavrasiya",
        ],
    },
    {
        "canonical": "Expressbank",
        "aliases": [
            "Expressbank", "Express Bank", "Expressbankın", "Express Bankı",
            "expressbank",
        ],
    },
    {
        "canonical": "Yelo Bank",
        "aliases": [
            "Yelo Bank", "YeloBank", "Yelo Bankı", "Yelo Bankın", "yelo.az",
        ],
    },
    {
        "canonical": "Premium Bank",
        "aliases": [
            "Premium Bank", "PremiumBank", "Premium Bankı", "Premium Bankın",
            "premiumbank",
        ],
    },
    {
        "canonical": "Rabitabank",
        "aliases": [
            "Rabitabank", "Rabitə Bank", "Rabitabankın", "Rabitə Bankı",
            "rabitabank",
        ],
    },
    {
        "canonical": "Unibank",
        "aliases": [
            "Unibank", "UnibankIN", "Unibankin", "Unibank", "unibank.az",
        ],
    },
    {
        "canonical": "VTB Azerbaijan",
        "aliases": [
            "VTB Azerbaijan", "VTB Azərbaycan", "VTB", "vtb.az",
        ],
    },
    {
        "canonical": "ASB",
        "aliases": [
            "ASB", "Azerbaijan Industry Bank", "Azərbaycan Sənaye Bankı",
            "asb.az",
        ],
    },
    {
        "canonical": "Azer-Turk Bank",
        "aliases": [
            "Azer-Turk Bank", "AzerTurk Bank", "ATB", "Azər-Türk Bank",
            "atb.az",
        ],
    },
    {
        "canonical": "Nakhchivanbank",
        "aliases": [
            "Nakhchivanbank", "Naxçıvanbank", "Naxçıvan Bank", "naxbank",
        ],
    },
    {
        "canonical": "TuranBank",
        "aliases": [
            "TuranBank", "Turan Bank", "Turan Bankı", "turanbank",
        ],
    },
    {
        "canonical": "MuganBank",
        "aliases": [
            "MuganBank", "Muğan Bank", "Mugan Bank", "Muğan Bankı",
            "muganbank",
        ],
    },
]


_AZ = str.maketrans({
    'ə':'e','Ə':'e','ö':'o','Ö':'o','ü':'u','Ü':'u',
    'ğ':'g','Ğ':'g','ş':'s','Ş':'s','ç':'c','Ç':'c','ı':'i','İ':'i',
})

def detect_bank(query: str) -> dict | None:
    """Return the matched bank entry if any alias appears in the query, else None.
    Normalizes Azerbaijani diacritics so 'pasa' matches 'PAŞA', etc."""
    q = query.translate(_AZ).lower()
    for bank in BANKS:
        for alias in bank["aliases"]:
            if alias.translate(_AZ).lower() in q:
                return bank
    return None
