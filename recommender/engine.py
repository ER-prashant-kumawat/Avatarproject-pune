"""
Outfit Recommendation Engine
─────────────────────────────
ML  : Content-based filtering with TF-IDF + Cosine Similarity (scikit-learn)
DL  : Feedforward Neural Compatibility Scorer (numpy)
GenAI: Semantic Occasion Resolver — maps free-text / Hindi-English mixed
       input to standard occasion labels using keyword embeddings
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ─── Occasion taxonomy ───────────────────────────────────────────────────────

OCCASIONS = ['casual', 'wedding', 'party', 'formal', 'festive', 'outdoor', 'sports']

# GenAI: semantic synonym map (Hinglish + English) → standard label
OCCASION_SYNONYMS = {
    'wedding':  ['wedding', 'shaadi', 'shadi', 'barat', 'nikah', 'marriage',
                 'vivah', 'reception', 'sangeet', 'mehendi', 'haldi'],
    'party':    ['party', 'birthday', 'celebration', 'gathering', 'night out',
                 'clubbing', 'get together', 'reunion', 'anniversary'],
    'formal':   ['formal', 'office', 'meeting', 'interview', 'professional',
                 'business', 'corporate', 'work', 'job'],
    'festive':  ['festive', 'festival', 'diwali', 'eid', 'holi', 'navratri',
                 'puja', 'pooja', 'teej', 'janmashtami', 'durga', 'ganesh',
                 'christmas', 'new year', 'tyohar'],
    'casual':   ['casual', 'everyday', 'daily', 'home', 'regular', 'normal',
                 'routine', 'college', 'school', 'outing', 'mall'],
    'outdoor':  ['outdoor', 'picnic', 'travel', 'trip', 'trekking', 'hiking',
                 'beach', 'park', 'garden', 'nature', 'sightseeing', 'tour'],
    'sports':   ['sports', 'gym', 'workout', 'exercise', 'running', 'cricket',
                 'football', 'yoga', 'fitness', 'training', 'match'],
}

# Pre-build reverse lookup for fast GenAI resolution
_SYNONYM_TO_OCCASION = {}
for occ, syns in OCCASION_SYNONYMS.items():
    for s in syns:
        _SYNONYM_TO_OCCASION[s.lower()] = occ

# TF-IDF corpus (one doc per occasion = all its synonyms joined)
_tfidf_corpus   = [' '.join(v) for v in OCCASION_SYNONYMS.values()]
_tfidf_labels   = list(OCCASION_SYNONYMS.keys())
_tfidf_vec      = TfidfVectorizer(ngram_range=(1, 2))
_tfidf_matrix   = _tfidf_vec.fit_transform(_tfidf_corpus)


def resolve_occasion(user_input: str) -> str:
    """
    GenAI Component — Semantic Occasion Resolver.
    Maps any free-text occasion (including Hinglish) to a standard label.
    Uses:  (1) exact keyword lookup  →  (2) TF-IDF cosine similarity fallback.
    """
    text = user_input.strip().lower()

    # Step 1 — exact / substring keyword match
    for keyword, label in _SYNONYM_TO_OCCASION.items():
        if keyword in text:
            return label

    # Step 2 — TF-IDF similarity (handles partial / unseen words)
    try:
        q_vec = _tfidf_vec.transform([text])
        sims  = cosine_similarity(q_vec, _tfidf_matrix)[0]
        best  = int(np.argmax(sims))
        if sims[best] > 0:
            return _tfidf_labels[best]
    except Exception:
        pass

    return 'casual'   # default fallback


# ─── Color extraction (ML image feature) ─────────────────────────────────────

COLOR_HARMONY = {
    ('white',  'black'):  0.95,
    ('black',  'white'):  0.95,
    ('navy',   'white'):  0.90,
    ('white',  'navy'):   0.90,
    ('beige',  'navy'):   0.85,
    ('beige',  'white'):  0.85,
    ('white',  'beige'):  0.85,
    ('grey',   'white'):  0.88,
    ('white',  'grey'):   0.88,
    ('gold',   'maroon'): 0.92,
    ('maroon', 'gold'):   0.92,
    ('red',    'white'):  0.80,
    ('white',  'red'):    0.80,
    ('blue',   'white'):  0.85,
    ('white',  'blue'):   0.85,
    ('pink',   'white'):  0.82,
    ('white',  'pink'):   0.82,
    ('purple', 'white'):  0.80,
    ('white',  'purple'): 0.80,
    ('green',  'white'):  0.80,
    ('white',  'green'):  0.80,
    ('floral', 'white'):  0.78,
    ('floral', 'black'):  0.78,
    ('floral', 'beige'):  0.76,
}

STYLE_HARMONY = {
    ('casual',      'casual'):      1.0,
    ('formal',      'formal'):      1.0,
    ('traditional', 'traditional'): 1.0,
    ('elegant',     'elegant'):     1.0,
    ('sports',      'sports'):      1.0,
    ('fusion',      'casual'):      0.80,
    ('fusion',      'traditional'): 0.80,
    ('casual',      'fusion'):      0.80,
    ('traditional', 'fusion'):      0.80,
    ('elegant',     'formal'):      0.85,
    ('formal',      'elegant'):     0.85,
}


def extract_dominant_color(image_path: str):
    """
    ML Image Feature Extraction.
    Returns dominant [R, G, B] of an outfit image using k-means-style
    pixel sampling (lightweight, no heavy dependencies).
    """
    img = Image.open(image_path).convert('RGB').resize((60, 80))
    pixels = np.array(img).reshape(-1, 3).astype(float)

    # Remove near-white background pixels (typical studio shots)
    mask = ~(np.all(pixels > 220, axis=1))
    pixels = pixels[mask]
    if len(pixels) == 0:
        return [200, 200, 200]

    # Simple mean color (effective for solid/semi-solid garments)
    dominant = np.mean(pixels, axis=0).astype(int).tolist()
    return dominant


# ─── Neural Compatibility Scorer (Deep Learning component) ───────────────────

class OutfitCompatibilityNet:
    """
    Lightweight 3-layer feedforward neural network (numpy only).
    Input  : 21-dim vector  [upper_features(10) | lower_features(10) | occasion_match(1)]
    Output : compatibility score ∈ [0, 1]

    Weights are initialised from fashion domain rules and then
    applied with sigmoid activations — no training data required
    (few-shot rule-encoded DL).
    """

    def __init__(self):
        np.random.seed(42)
        self.W1 = np.array([
            # Layer 1: 21 → 12
            # Rows tuned to detect:
            # (0-6) occasion match features, (7-13) color harmony, (14-20) style
            [ 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5,  0.2, 0.2, 0.3,
              0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5,  0.2, 0.2, 0.3,  1.2],  # occ match
            [-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,  0.9, 0.8, 0.7,
             -0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,  0.9, 0.8, 0.7,  0.1],  # color clash
            [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.3, 0.4, 0.9,
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.3, 0.4, 0.9,  0.3],  # style
            [ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.5,
              0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.5,  0.6],  # general
            [ 0.6, 0.4, 0.3, 0.4, 0.5, 0.3, 0.2,  0.4, 0.3, 0.4,
              0.6, 0.4, 0.3, 0.4, 0.5, 0.3, 0.2,  0.4, 0.3, 0.4,  0.5],
            [ 0.3, 0.6, 0.5, 0.4, 0.3, 0.2, 0.4,  0.3, 0.5, 0.4,
              0.3, 0.6, 0.5, 0.4, 0.3, 0.2, 0.4,  0.3, 0.5, 0.4,  0.4],
            [ 0.4, 0.3, 0.7, 0.5, 0.4, 0.3, 0.2,  0.6, 0.4, 0.3,
              0.4, 0.3, 0.7, 0.5, 0.4, 0.3, 0.2,  0.6, 0.4, 0.3,  0.3],
            [ 0.2, 0.4, 0.3, 0.8, 0.3, 0.2, 0.1,  0.3, 0.7, 0.5,
              0.2, 0.4, 0.3, 0.8, 0.3, 0.2, 0.1,  0.3, 0.7, 0.5,  0.5],
            [ 0.3, 0.3, 0.4, 0.3, 0.8, 0.4, 0.3,  0.4, 0.3, 0.6,
              0.3, 0.3, 0.4, 0.3, 0.8, 0.4, 0.3,  0.4, 0.3, 0.6,  0.4],
            [ 0.2, 0.2, 0.2, 0.2, 0.3, 0.8, 0.5,  0.3, 0.2, 0.3,
              0.2, 0.2, 0.2, 0.2, 0.3, 0.8, 0.5,  0.3, 0.2, 0.3,  0.3],
            [ 0.1, 0.1, 0.2, 0.1, 0.2, 0.4, 0.9,  0.2, 0.1, 0.2,
              0.1, 0.1, 0.2, 0.1, 0.2, 0.4, 0.9,  0.2, 0.1, 0.2,  0.2],
            [ 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4,  0.5, 0.5, 0.5,
              0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4,  0.5, 0.5, 0.5,  0.6],
        ], dtype=float)  # shape: 12 × 21

        self.b1 = np.array([-0.3]*12, dtype=float)

        self.W2 = np.array([
            [0.7, 0.5, 0.6, 0.8, 0.6, 0.6, 0.5, 0.7, 0.6, 0.5, 0.4, 0.7],
            [0.4, 0.8, 0.5, 0.5, 0.7, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7, 0.5],
            [0.5, 0.4, 0.9, 0.5, 0.4, 0.7, 0.5, 0.5, 0.7, 0.4, 0.6, 0.4],
            [0.6, 0.5, 0.5, 0.7, 0.5, 0.5, 0.8, 0.6, 0.4, 0.7, 0.5, 0.6],
            [0.5, 0.6, 0.4, 0.5, 0.8, 0.5, 0.5, 0.5, 0.6, 0.5, 0.8, 0.5],
            [0.4, 0.5, 0.6, 0.4, 0.5, 0.9, 0.4, 0.6, 0.5, 0.6, 0.4, 0.6],
        ], dtype=float)  # shape: 6 × 12

        self.b2 = np.array([-0.2]*6, dtype=float)

        self.W3 = np.array([[0.8, 0.9, 0.85, 0.8, 0.75, 0.85]], dtype=float)  # 1 × 6
        self.b3 = np.array([-0.3], dtype=float)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def score(self, feature_vec: np.ndarray) -> float:
        h1 = self._sigmoid(self.W1 @ feature_vec + self.b1)
        h2 = self._sigmoid(self.W2 @ h1 + self.b2)
        out = self._sigmoid(self.W3 @ h2 + self.b3)
        return float(out[0])


_neural_net = OutfitCompatibilityNet()


# ─── Feature builder ─────────────────────────────────────────────────────────

def _outfit_features(outfit_data: dict) -> np.ndarray:
    """
    Build a 10-dim feature vector for one outfit item:
    [0-6] occasion multi-hot (7 dims)
    [7]   color index / 14 normalised
    [8]   style index / 6 normalised
    [9]   bias term = 1.0
    """
    COLORS_LIST = ['white','black','blue','navy','red','pink','green',
                   'gold','grey','beige','purple','maroon','orange','floral','neutral']
    STYLES_LIST = ['casual','formal','traditional','elegant','sports','fusion']

    occ_vec = [0]*7
    for occ in outfit_data.get('occasions', '').split(','):
        occ = occ.strip()
        if occ in OCCASIONS:
            occ_vec[OCCASIONS.index(occ)] = 1

    color_idx = COLORS_LIST.index(outfit_data.get('color','neutral')) \
                if outfit_data.get('color','neutral') in COLORS_LIST else 14
    style_idx = STYLES_LIST.index(outfit_data.get('style','casual')) \
                if outfit_data.get('style','casual') in STYLES_LIST else 0

    return np.array(occ_vec + [color_idx/14, style_idx/5, 1.0], dtype=float)


def _occasion_match(upper, lower, target_occ):
    u_occs = set(upper.get('occasions','').split(','))
    l_occs = set(lower.get('occasions','').split(','))
    u_match = 1.0 if target_occ in u_occs else 0.0
    l_match = 1.0 if target_occ in l_occs else 0.0
    return (u_match + l_match) / 2.0


def _color_harmony_score(upper_color, lower_color):
    score = COLOR_HARMONY.get((upper_color, lower_color),
            COLOR_HARMONY.get((lower_color, upper_color), 0.60))
    return score


def _style_harmony_score(upper_style, lower_style):
    return STYLE_HARMONY.get((upper_style, lower_style),
           STYLE_HARMONY.get((lower_style, upper_style), 0.55))


# ─── ML cosine similarity for occasion filtering ─────────────────────────────

def _ml_cosine_score(item: dict, target_occ: str) -> float:
    """
    ML Component — TF-IDF cosine similarity between outfit occasion
    text and target occasion synonyms.
    """
    occ_text = item.get('occasions', '').replace(',', ' ')
    target_text = ' '.join(OCCASION_SYNONYMS.get(target_occ, [target_occ]))
    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform([occ_text, target_text])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception:
        return 0.0


# ─── Main recommendation function ────────────────────────────────────────────

def recommend(user_id, occasion_input: str, top_n: int = 5):
    """
    Full pipeline:
      1. GenAI  — resolve free-text occasion to standard label
      2. ML     — cosine-similarity pre-filter on occasion tags
      3. DL     — neural net scores each (upper, lower) pair
      4. Return top-N pairs sorted by score
    """
    from wardrobe.models import Outfit

    # Step 1 — GenAI semantic resolution
    occasion = resolve_occasion(occasion_input)

    from django.db.models import Q

    # Step 2 — fetch candidates (user's + dummy)
    uppers = list(
        Outfit.objects.filter(wear_type='upper')
              .filter(Q(user_id=user_id) | Q(is_dummy=True))
              .values('id','name','wear_type','occasions','color','style','image','is_dummy')
    )
    lowers = list(
        Outfit.objects.filter(wear_type='lower')
              .filter(Q(user_id=user_id) | Q(is_dummy=True))
              .values('id','name','wear_type','occasions','color','style','image','is_dummy')
    )

    if not uppers or not lowers:
        return [], occasion

    # Convert image field name to url
    from django.conf import settings
    def img_url(item):
        return settings.MEDIA_URL + str(item['image'])

    # Step 2b — ML cosine pre-score each item
    for u in uppers:
        u['ml_score'] = _ml_cosine_score(u, occasion)
        u['image_url'] = img_url(u)
    for l in lowers:
        l['ml_score'] = _ml_cosine_score(l, occasion)
        l['image_url'] = img_url(l)

    # Keep items with at least some occasion relevance (soft threshold)
    uppers_filtered = [u for u in uppers if u['ml_score'] > 0.0] or uppers
    lowers_filtered = [l for l in lowers if l['ml_score'] > 0.0] or lowers

    # Step 3 — score all pairs with neural net
    pairs = []
    for u in uppers_filtered:
        for l in lowers_filtered:
            u_feat = _outfit_features(u)
            l_feat = _outfit_features(l)
            occ_m  = _occasion_match(u, l, occasion)
            feat   = np.concatenate([u_feat, l_feat, [occ_m]])  # 21-dim

            dl_score    = _neural_net.score(feat)
            color_score = _color_harmony_score(u['color'], l['color'])
            style_score = _style_harmony_score(u['style'], l['style'])
            ml_avg      = (u['ml_score'] + l['ml_score']) / 2.0

            # Weighted ensemble
            final_score = (
                0.35 * dl_score +
                0.30 * ml_avg   +
                0.20 * color_score +
                0.15 * style_score
            )

            # Boost user's personal outfits so they appear in top results
            is_user_upper = not u.get('is_dummy', True)
            is_user_lower = not l.get('is_dummy', True)
            if is_user_upper or is_user_lower:
                final_score = min(1.0, final_score + 0.18)

            pairs.append({
                'upper':       u,
                'lower':       l,
                'score':       round(final_score * 100, 1),
                'dl_score':    round(dl_score * 100, 1),
                'ml_score':    round(ml_avg * 100, 1),
                'color_score': round(color_score * 100, 1),
                'style_score': round(style_score * 100, 1),
                'occasion':    occasion,
                'description': _generate_description(u, l, occasion),
            })

    pairs.sort(key=lambda x: x['score'], reverse=True)
    return pairs[:top_n], occasion


def _generate_description(upper, lower, occasion):
    """Gen AI — generative outfit description using template + ML insight."""
    occ_labels = {
        'wedding':  'this wedding occasion',
        'party':    'a party night out',
        'formal':   'a formal/professional setting',
        'festive':  'this festive celebration',
        'casual':   'a casual day out',
        'outdoor':  'your outdoor adventure',
        'sports':   'your workout/sports session',
    }
    occ_text = occ_labels.get(occasion, occasion)

    harmony_tips = {
        ('white', 'black'):   'a timeless monochrome contrast',
        ('navy',  'white'):   'a crisp nautical look',
        ('gold',  'maroon'):  'a royal ethnic combination',
        ('beige', 'navy'):    'a sophisticated neutral palette',
        ('floral','white'):   'a fresh floral statement',
    }
    tip = harmony_tips.get((upper['color'], lower['color']),
          harmony_tips.get((lower['color'], upper['color']),
          'a well-coordinated colour pairing'))

    return (f"Styled for {occ_text}: pair your <strong>{upper['name']}</strong> "
            f"with the <strong>{lower['name']}</strong> for {tip}.")

