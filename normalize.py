"""
Text normalization for Indic TTS input.

MMS-TTS/VITS tokenizers here have tiny (58-84 token) grapheme-level
vocabularies -- essentially just the native script's characters. Digits,
Latin letters, and most symbols are OUT of vocabulary and get silently
dropped or mispronounced. Normalization converts them into native-script
words the model can actually pronounce.

Number-to-words: implemented as digit-by-digit spelling (each digit named
individually) rather than natural "twenty-five"-style grammar. This is a
deliberate scope choice -- natural number grammar (place values, the
Indian lakh/crore numbering convention) differs per language and needs
native-speaker-verified rules to get right; digit-by-digit is unambiguous
and correct for all 11 languages today. Swap in proper per-language
number grammar as a follow-up if natural-sounding numbers matter more
than shipping now.
"""
import re

DIGIT_WORDS = {
    "hin": ["शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ"],
    "mar": ["शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ"],
    "tam": ["பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"],
    "tel": ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"],
    "mal": ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒൻപത്"],
    "kan": ["ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು"],
    "guj": ["શૂન્ય", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ"],
    "ben": ["শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"],
    "pan": ["ਸਿਫ਼ਰ", "ਇੱਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ"],
    "ory": ["ଶୂନ୍ୟ", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ"],
    "asm": ["শূন্য", "এক", "দুই", "তিনি", "চাৰি", "পাঁচ", "ছয়", "সাত", "আঠ", "ন"],
}

_NUMBER_RE = re.compile(r"\d+")


def normalize_numbers(text: str, lang: str) -> str:
    words = DIGIT_WORDS.get(lang, DIGIT_WORDS["hin"])

    def _replace(match: re.Match) -> str:
        digits = match.group(0)
        return " ".join(words[int(d)] for d in digits)

    return _NUMBER_RE.sub(_replace, text)


# Symbols with no native pronunciation in the tiny grapheme vocab -- either
# drop them or replace with a native-script equivalent that IS in vocab.
_SYMBOL_STRIP = re.compile(r"[#*_~`^\[\]{}<>|\\@]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lang: str) -> str:
    text = normalize_numbers(text, lang)
    text = _SYMBOL_STRIP.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
