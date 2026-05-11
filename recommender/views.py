from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
import json
import os
import sys
import io
import base64
from .engine import recommend, resolve_occasion
from django.conf import settings

# HuggingFace token — loaded once at import time
_HF_TOKEN = getattr(settings, 'HUGGINGFACE_API_TOKEN', None)

# ── TripoSR local path ─────────────────────────────────────────────────────────
_TRIPOSR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TripoSR'
)
if os.path.isdir(_TRIPOSR_DIR) and _TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, _TRIPOSR_DIR)

# Cached singletons (loaded lazily on first use)
_TRIPOSR_MODEL  = None
_TRIPOSR_DEVICE = None
_REMBG_SESSION  = None


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        import rembg
        _REMBG_SESSION = rembg.new_session()
    return _REMBG_SESSION


def _rembg_remove(img_pil):
    """AI-based background removal via rembg — much cleaner than flood-fill."""
    import rembg
    return rembg.remove(img_pil, session=_get_rembg_session())


def _get_triposr_model():
    """Load TripoSR model once and cache it in memory."""
    global _TRIPOSR_MODEL, _TRIPOSR_DEVICE
    if _TRIPOSR_MODEL is not None:
        return _TRIPOSR_MODEL, _TRIPOSR_DEVICE
    import torch
    from tsr.system import TSR
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = TSR.from_pretrained(
        'stabilityai/TripoSR',
        config_name='config.yaml',
        weight_name='model.ckpt',
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    _TRIPOSR_MODEL  = model
    _TRIPOSR_DEVICE = device
    return model, device


@login_required
@require_POST
def get_recommendations(request):
    try:
        data = json.loads(request.body)
    except Exception:
        data = {}

    occasion_input = data.get('occasion', 'casual')
    top_n = int(data.get('top_n', 5))

    pairs, resolved_occasion = recommend(request.user.id, occasion_input, top_n)

    results = []
    for p in pairs:
        results.append({
            'upper': {
                'id':        p['upper']['id'],
                'name':      p['upper']['name'],
                'image_url': p['upper']['image_url'],
                'color':     p['upper']['color'],
                'style':     p['upper']['style'],
                'occasions': p['upper']['occasions'],
                'is_dummy':  p['upper'].get('is_dummy', False),
            },
            'lower': {
                'id':        p['lower']['id'],
                'name':      p['lower']['name'],
                'image_url': p['lower']['image_url'],
                'color':     p['lower']['color'],
                'style':     p['lower']['style'],
                'occasions': p['lower']['occasions'],
                'is_dummy':  p['lower'].get('is_dummy', False),
            },
            'score':             p['score'],
            'dl_score':          p['dl_score'],
            'ml_score':          p['ml_score'],
            'color_score':       p['color_score'],
            'style_score':       p['style_score'],
            'description':       p['description'],
            'resolved_occasion': resolved_occasion,
        })

    return JsonResponse({
        'ok':      True,
        'results': results,
        'occasion': resolved_occasion,
    })


def _remove_background(img_rgba, tolerance=45):
    """
    Remove white/light/grey/beige background from garment image using flood-fill.
    """
    import numpy as np
    from collections import deque
    from PIL import Image as PILImage, ImageFilter

    data = np.array(img_rgba, dtype=np.int32)
    h, w = data.shape[:2]
    alpha = data[:, :, 3].copy()
    visited = np.zeros((h, w), dtype=bool)

    def is_bg(r, g, b):
        brightness = (r + g + b) / 3
        variation = max(abs(r - g), abs(g - b), abs(r - b))
        # Pure white / near-white
        if r > (255 - tolerance) and g > (255 - tolerance) and b > (255 - tolerance):
            return True
        # Light grey or very low-saturation light color
        if brightness > 210 and variation < 25:
            return True
        return False

    # Seed flood-fill from all 4 edges
    queue = deque()
    for x in range(w):
        for y in [0, h - 1]:
            if not visited[y, x]:
                r, g, b = int(data[y, x, 0]), int(data[y, x, 1]), int(data[y, x, 2])
                if is_bg(r, g, b):
                    visited[y, x] = True
                    queue.append((y, x))
    for y in range(h):
        for x in [0, w - 1]:
            if not visited[y, x]:
                r, g, b = int(data[y, x, 0]), int(data[y, x, 1]), int(data[y, x, 2])
                if is_bg(r, g, b):
                    visited[y, x] = True
                    queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        alpha[cy, cx] = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                r, g, b = int(data[ny, nx, 0]), int(data[ny, nx, 1]), int(data[ny, nx, 2])
                if is_bg(r, g, b):
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    # Smooth edges
    alpha_img = PILImage.fromarray(alpha.astype(np.uint8), mode='L')
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=1))
    alpha_smooth = np.array(alpha_img)
    data[:, :, 3] = np.minimum(alpha_smooth, alpha)
    return PILImage.fromarray(data.astype(np.uint8), 'RGBA')


def _autocrop_transparent(img_rgba, padding=6):
    """Crop out empty transparent border from garment after bg removal."""
    import numpy as np
    data = np.array(img_rgba)
    alpha = data[:, :, 3]
    rows = np.any(alpha > 15, axis=1)
    cols = np.any(alpha > 15, axis=0)
    if not rows.any() or not cols.any():
        return img_rgba
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = data.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return img_rgba.crop((cmin, rmin, cmax + 1, rmax + 1))


def _pil_composite_tryon(person_path, upper_path, lower_path):
    """
    Show person photo with outfit clothes adjusted on correct body regions.
    Strategy:
      1. Draw mannequin body (always visible, clear body shape)
      2. Place person's face on mannequin head (personal touch)
      3. Overlay upper garment on torso region
      4. Overlay lower garment on leg region
    This guarantees avatar (body) is always visible with clothes on top.
    """
    from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

    # Use the mannequin body as base — it's ALWAYS visible, no body hidden under clothes
    W, H = 480, 720
    SKIN  = (220, 175, 145)
    SHOE  = (40, 32, 28)
    BG    = (238, 238, 243)
    cx    = W // 2

    canvas = Image.new('RGBA', (W, H), BG + (255,))
    draw   = ImageDraw.Draw(canvas)

    # ── Head ──────────────────────────────────────────────────────────────────
    hr = int(W * 0.10)
    hy = int(H * 0.105)
    draw.ellipse([cx-hr, hy-hr, cx+int(hr*0.98), hy+hr], fill=SKIN+(255,))

    # ── Neck ──────────────────────────────────────────────────────────────────
    nw  = int(W * 0.048)
    nbot = int(H * 0.220)
    draw.rectangle([cx-nw, hy+hr-6, cx+nw, nbot], fill=SKIN+(255,))

    # ── Torso (tapered: wide shoulders, narrow waist) ─────────────────────────
    sh_w  = int(W * 0.22)
    wst_w = int(W * 0.14)
    t_top = int(H * 0.215)
    t_bot = int(H * 0.545)
    draw.polygon([
        (cx-sh_w, t_top), (cx+sh_w, t_top),
        (cx+wst_w, t_bot), (cx-wst_w, t_bot),
    ], fill=SKIN+(255,))

    # ── Arms ──────────────────────────────────────────────────────────────────
    atw, abw = int(W*0.065), int(W*0.050)
    aty, aby = int(H*0.228), int(H*0.548)
    lax = cx - sh_w - 2
    draw.polygon([(lax,aty),(lax-atw,aty+int((aby-aty)*.05)),(lax-abw,aby),(lax+4,aby)], fill=SKIN+(255,))
    rax = cx + sh_w + 2
    draw.polygon([(rax,aty),(rax+atw,aty+int((aby-aty)*.05)),(rax+abw,aby),(rax-4,aby)], fill=SKIN+(255,))

    # ── Hips ──────────────────────────────────────────────────────────────────
    hp_w  = int(W * 0.168)
    hp_bot = int(H * 0.585)
    draw.polygon([(cx-wst_w,t_bot),(cx+wst_w,t_bot),(cx+hp_w,hp_bot),(cx-hp_w,hp_bot)], fill=SKIN+(255,))

    # ── Legs ──────────────────────────────────────────────────────────────────
    lw   = int(W * 0.105)
    lgap = int(W * 0.020)
    lt   = int(H * 0.578)
    lb   = int(H * 0.930)
    tp   = int(lw * 0.12)
    draw.polygon([(cx-lgap-lw,lt),(cx-lgap,lt),(cx-lgap-tp,lb),(cx-lgap-lw-tp,lb)], fill=SKIN+(255,))
    draw.polygon([(cx+lgap,lt),(cx+lgap+lw,lt),(cx+lgap+lw+tp,lb),(cx+lgap+tp,lb)], fill=SKIN+(255,))

    # ── Shoes ─────────────────────────────────────────────────────────────────
    draw.ellipse([cx-lgap-lw-tp-8, int(H*.918), cx-lgap-tp+14,   int(H*.958)], fill=SHOE+(255,))
    draw.ellipse([cx+lgap+tp-14,   int(H*.918), cx+lgap+lw+tp+8, int(H*.958)], fill=SHOE+(255,))

    # ── Upper garment (shirt) over torso + arms ────────────────────────────────
    if upper_path and os.path.isfile(upper_path):
        try:
            shirt = Image.open(upper_path).convert('RGBA')
            shirt = _remove_background(shirt, tolerance=55)
            shirt = _autocrop_transparent(shirt, padding=5)
            gw, gh = shirt.size
            tw = int(W * 0.92)
            th = int(gh * tw / max(gw, 1))
            max_h = int(H * 0.36)
            if th > max_h:
                th = max_h
                tw = int(gw * th / max(gh, 1))
            shirt = shirt.resize((tw, th), Image.LANCZOS)
            # Enhance colours
            rgb = ImageEnhance.Contrast(shirt.convert('RGB')).enhance(1.1)
            r, g_ch, b_ch = rgb.split()
            _, _, _, a = shirt.split()
            shirt = Image.merge('RGBA', (r, g_ch, b_ch, a))
            canvas.paste(shirt, ((W-tw)//2, int(H*0.210)), shirt)
        except Exception:
            pass

    # ── Lower garment (pants/skirt) over hips + legs ──────────────────────────
    if lower_path and os.path.isfile(lower_path):
        try:
            pants = Image.open(lower_path).convert('RGBA')
            pants = _remove_background(pants, tolerance=55)
            pants = _autocrop_transparent(pants, padding=5)
            gw, gh = pants.size
            tw = int(W * 0.64)
            th = int(gh * tw / max(gw, 1))
            max_h = int(H * 0.44)
            if th > max_h:
                th = max_h
                tw = int(gw * th / max(gh, 1))
            pants = pants.resize((tw, th), Image.LANCZOS)
            rgb = ImageEnhance.Contrast(pants.convert('RGB')).enhance(1.08)
            r, g_ch, b_ch = rgb.split()
            _, _, _, a = pants.split()
            pants = Image.merge('RGBA', (r, g_ch, b_ch, a))
            canvas.paste(pants, ((W-tw)//2, int(H*0.530)), pants)
        except Exception:
            pass

    # ── Person face on head (circular crop from top-centre of photo) ──────────
    if person_path and os.path.isfile(person_path):
        try:
            face = Image.open(person_path).convert('RGBA')
            fw, fh = face.size
            crop_sz = min(fw, int(fh * 0.42))
            left    = (fw - crop_sz) // 2
            face    = face.crop((left, 0, left + crop_sz, crop_sz))
            fs      = (hr * 2) - 6
            face    = face.resize((fs, fs), Image.LANCZOS)
            mask    = Image.new('L', (fs, fs), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, fs, fs], fill=255)
            mask    = mask.filter(ImageFilter.GaussianBlur(radius=1.5))
            face.putalpha(mask)
            canvas.paste(face, (cx - fs//2, hy - fs//2), face)
        except Exception:
            pass

    result = canvas.convert('RGB')
    buf = io.BytesIO()
    result.save(buf, format='JPEG', quality=94)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


def _apply_lower_on_b64(img_b64, lower_path):
    """
    Overlay lower garment on a HuggingFace result image (base64) using PIL.
    Returns updated base64 JPEG.
    """
    from PIL import Image, ImageEnhance

    header, data = img_b64.split(',', 1)
    img_bytes = base64.b64decode(data)
    base_img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    W, H = base_img.size
    result = base_img.copy()

    garment = Image.open(lower_path).convert('RGBA')
    garment = _remove_background(garment, tolerance=45)
    garment = _autocrop_transparent(garment, padding=4)

    # Lower body: waist (50%) to feet (100%) — covers original jeans too
    region_w = int(W * 0.78)
    region_h = int(H * 0.50)
    gw, gh = garment.size
    scale = min(region_w / gw, region_h / gh)
    garment = garment.resize((int(gw * scale), int(gh * scale)), Image.LANCZOS)

    rgb = garment.convert('RGB')
    rgb = ImageEnhance.Brightness(rgb).enhance(1.04)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
    r, g, b = rgb.split()
    _, _, _, a = garment.split()
    a = a.point(lambda v: int(v * 0.97))
    garment = Image.merge('RGBA', (r, g, b, a))

    new_gw, new_gh = garment.size
    px = (W - new_gw) // 2
    py = int(H * 0.50)          # start from waist, not mid-leg
    result.paste(garment, (px, py), garment)

    result_rgb = result.convert('RGB')
    buf = io.BytesIO()
    result_rgb.save(buf, format='JPEG', quality=93)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


def _read_img_b64(path, fmt='PNG'):
    with open(path, 'rb') as f:
        return f'data:image/{fmt.lower()};base64,{base64.b64encode(f.read()).decode()}'


def _try_huggingface(person_path, garment_path, garment_desc):
    """
    Try all HuggingFace spaces in priority order.
    Priority:
      1. yisol/IDM-VTON          — best virtual try-on
      2. Nymbo/Virtual-Try-ON    — fast try-on
      3. TencentARC/PhotoMaker   — face-preserving, prompt-based
      4. InstantX/InstantID      — face identity + outfit prompt
      5. h94/IP-Adapter-FaceID   — IP-Adapter face-preserving
      6. modelscope/facechain    — realistic avatar
    Returns (base64_image_string, space_id)
    """
    from gradio_client import Client, handle_file

    outfit_prompt = (
        f'a realistic full body photo of a person wearing {garment_desc}, '
        f'photorealistic, sharp focus, studio lighting, white background'
    )
    neg_prompt = (
        'nude, nsfw, blurry, low quality, watermark, duplicate, '
        'extra limbs, bad anatomy, deformed'
    )

    last_err = None

    # ── 1. IDM-VTON ──────────────────────────────────────────────────────────
    try:
        client = Client('yisol/IDM-VTON', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            dict={"background": handle_file(person_path), "layers": [], "composite": None},
            garm_img=handle_file(garment_path),
            garment_des=garment_desc,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=20,
            seed=42,
            api_name='/tryon',
        )
        return _read_img_b64(result[0]), 'yisol/IDM-VTON'
    except Exception as e:
        last_err = str(e)

    # ── 2. Nymbo Virtual-Try-ON ───────────────────────────────────────────────
    try:
        client = Client('Nymbo/Virtual-Try-ON', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            dict={"background": handle_file(person_path), "layers": [], "composite": None},
            garm_img=handle_file(garment_path),
            garment_des=garment_desc,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name='/tryon',
        )
        return _read_img_b64(result[0]), 'Nymbo/Virtual-Try-ON'
    except Exception as e:
        last_err = str(e)

    # ── 3. PhotoMaker ─────────────────────────────────────────────────────────
    try:
        client = Client('TencentARC/PhotoMaker', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            upload_images=[handle_file(person_path)],
            style_strength_ratio=20,
            prompt=outfit_prompt,
            negative_prompt=neg_prompt,
            num_steps=50,
            style_name='Photographic (Default)',
            num_outputs=1,
            guidance_scale=5,
            seed=42,
            api_name='/generate_image',
        )
        out = result[0]
        img_path = out[0]['image'] if isinstance(out, list) and isinstance(out[0], dict) else out
        return _read_img_b64(img_path), 'TencentARC/PhotoMaker'
    except Exception as e:
        last_err = str(e)

    # ── 4. InstantID ──────────────────────────────────────────────────────────
    try:
        client = Client('InstantX/InstantID', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            face_file=handle_file(person_path),
            prompt=outfit_prompt,
            negative_prompt=neg_prompt,
            style='(No style)',
            num_steps=30,
            guidance_scale=5,
            seed=42,
            api_name='/generate_image',
        )
        img_path = result[0] if isinstance(result, (list, tuple)) else result
        return _read_img_b64(img_path), 'InstantX/InstantID'
    except Exception as e:
        last_err = str(e)

    # ── 5. IP-Adapter FaceID ──────────────────────────────────────────────────
    try:
        client = Client('h94/IP-Adapter-FaceID', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            images=[handle_file(person_path)],
            prompt=outfit_prompt,
            negative_prompt=neg_prompt,
            scale=0.8,
            num_samples=1,
            seed=42,
            num_inference_steps=30,
            api_name='/generate',
        )
        img_path = result[0] if isinstance(result, (list, tuple)) else result
        return _read_img_b64(img_path), 'h94/IP-Adapter-FaceID'
    except Exception as e:
        last_err = str(e)

    # ── 6. FaceChain ──────────────────────────────────────────────────────────
    try:
        client = Client('modelscope/facechain', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            user_face=handle_file(person_path),
            style_prompt=outfit_prompt,
            neg_prompt=neg_prompt,
            num_generate=1,
            multiplier_style=0.25,
            api_name='/generate',
        )
        img_path = result[0] if isinstance(result, (list, tuple)) else result
        return _read_img_b64(img_path), 'modelscope/facechain'
    except Exception as e:
        last_err = str(e)

    raise RuntimeError(last_err or 'All HuggingFace spaces failed.')


@login_required
@require_POST
def try_on_view(request):
    """
    AI Virtual Try-On.
    Tries HuggingFace spaces first; falls back to PIL composite overlay.
    """
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)

    upper_url    = data.get('upper_url', '') or data.get('garment_url', '')
    lower_url    = data.get('lower_url', '')
    garment_desc = data.get('garment_desc', 'outfit')

    if not request.user.profile_photo:
        return JsonResponse({
            'ok': False,
            'error': 'Please upload your full-body photo first (click "Add Photo" on the avatar).'
        }, status=400)

    person_path = request.user.profile_photo.path
    if not os.path.isfile(person_path):
        return JsonResponse({'ok': False, 'error': 'Profile photo file missing. Please re-upload.'}, status=400)

    def resolve_path(url):
        if not url:
            return None
        rel = url.replace(settings.MEDIA_URL, '').lstrip('/')
        p   = os.path.join(settings.MEDIA_ROOT, rel)
        return p if os.path.isfile(p) else None

    upper_path = resolve_path(upper_url)
    lower_path = resolve_path(lower_url)

    if not upper_path and not lower_path:
        return JsonResponse({'ok': False, 'error': 'Outfit image not found.'}, status=400)

    # 1. Try HuggingFace AI first (IDM-VTON — photorealistic person wearing clothes)
    hf_tried = False
    if upper_path:
        try:
            img_data, space = _try_huggingface(person_path, upper_path, garment_desc)
            if lower_path:
                img_data = _apply_lower_on_b64(img_data, lower_path)
            return JsonResponse({'ok': True, 'image': img_data, 'method': f'AI ({space})'})
        except Exception:
            hf_tried = True

    # 2. PIL composite tryon — person photo with clothes overlaid at body regions
    #    Shows actual person wearing the outfit (no floating clothes)
    try:
        img_data = _pil_composite_tryon(person_path, upper_path, lower_path)
        note = ' (HuggingFace unavailable — showing local overlay)' if hf_tried else ''
        return JsonResponse({'ok': True, 'image': img_data, 'method': 'StyleSense Smart Overlay'})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'Try-on failed: {str(e)[:150]}'}, status=500)


# ── 3D Avatar helpers ──────────────────────────────────────────────────────────

def _hf_inference_img(model_id, prompt, neg_prompt='', width=512, height=768, steps=25):
    """
    Call HuggingFace Inference API for text-to-image.
    Returns raw image bytes or raises.
    """
    import requests as req
    url  = f'https://api-inference.huggingface.co/models/{model_id}'
    hdrs = {'Authorization': f'Bearer {_HF_TOKEN}', 'Content-Type': 'application/json'}
    body = {
        'inputs': prompt,
        'parameters': {
            'negative_prompt': neg_prompt,
            'width': width,
            'height': height,
            'num_inference_steps': steps,
            'guidance_scale': 7.5,
        }
    }
    resp = req.post(url, headers=hdrs, json=body, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f'HF API {resp.status_code}: {resp.text[:200]}')
    if resp.headers.get('content-type', '').startswith('application/json'):
        raise RuntimeError(f'Model loading: {resp.text[:200]}')
    return resp.content   # raw PNG/JPEG bytes


def _try_avatar3d_huggingface(person_path, outfit_desc):
    """
    Generate 3D avatar using HuggingFace models.
    Priority:
      1. TripoSR      (stabilityai) — single image → 3D render, 5-10s
      2. InstantMesh  (TencentARC)  — single image → high quality 3D mesh
      3. Zero123++    (sudo-ai)     — single image → multi-view 3D
      4. SDXL         (stabilityai) — text+image → game character render
      5. FLUX.1-dev                 — best text → image quality
    Returns (base64_img_str, model_name) or raises.
    """
    from gradio_client import Client, handle_file

    outfit_txt  = outfit_desc or 'casual stylish outfit'
    GAME_PROMPT = (
        f'3D game character, full body, {outfit_txt}, '
        f'Unreal Engine 5 render, photorealistic 3D model, '
        f'light grey background, standing pose, detailed clothing, '
        f'realistic proportions, sharp focus, studio lighting'
    )
    NEG = (
        'blurry, watermark, low quality, deformed, bad anatomy, nude, nsfw, '
        'extra limbs, missing limbs, disfigured, ugly, flat, 2D, cartoon'
    )
    last_err = None

    def _extract_image_from_result(result):
        """Try to get an image path from gradio result (handles list/dict/str)."""
        if isinstance(result, str) and os.path.isfile(result):
            return result
        if isinstance(result, (list, tuple)):
            for item in result:
                path = _extract_image_from_result(item)
                if path:
                    return path
        if isinstance(result, dict):
            for key in ('image', 'url', 'path', 'filepath'):
                if key in result:
                    v = result[key]
                    if isinstance(v, str) and os.path.isfile(v):
                        return v
        return None

    # ── 1. TripoSR (Stability AI) — image → 3D render (5-10s) ───────────────
    try:
        client = Client('stabilityai/TripoSR', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            handle_file(person_path),
            True,   # remove_background
            0.85,   # foreground_ratio
            api_name='/generate',
        )
        img_path = _extract_image_from_result(result)
        if img_path:
            return _read_img_b64(img_path), 'TripoSR 3D'
    except Exception as e:
        last_err = f'TripoSR: {e}'

    # ── 2. InstantMesh (TencentARC) — image → mesh render ───────────────────
    try:
        client = Client('TencentARC/InstantMesh', hf_token=_HF_TOKEN, verbose=False)
        # Step 1: preprocess image
        proc = client.predict(
            handle_file(person_path),
            True,
            api_name='/check_input_image',
        )
        proc_path = _extract_image_from_result(proc) or person_path
        # Step 2: generate 3D
        result = client.predict(
            handle_file(proc_path),
            42,     # seed
            api_name='/make3d',
        )
        img_path = _extract_image_from_result(result)
        if img_path:
            return _read_img_b64(img_path), 'InstantMesh 3D'
    except Exception as e:
        last_err = f'InstantMesh: {e}'

    # ── 3. Zero123++ — image → multi-view 3D (pick front view) ──────────────
    try:
        client = Client('sudo-ai/zero123plus', hf_token=_HF_TOKEN, verbose=False)
        result = client.predict(
            handle_file(person_path),
            api_name='/predict',
        )
        img_path = _extract_image_from_result(result)
        if img_path:
            # Zero123++ returns a grid of 6 views — crop the top-left (front)
            from PIL import Image as PILImage
            grid = PILImage.open(img_path).convert('RGB')
            gw, gh = grid.size
            front = grid.crop((0, 0, gw // 3, gh // 2))
            buf = io.BytesIO()
            front.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f'data:image/png;base64,{b64}', 'Zero123++ 3D'
    except Exception as e:
        last_err = f'Zero123++: {e}'

    # ── 4. SDXL (Stability AI) — text → game character ───────────────────────
    try:
        img_bytes = _hf_inference_img(
            'stabilityai/stable-diffusion-xl-base-1.0',
            GAME_PROMPT, NEG, width=512, height=768, steps=20
        )
        b64 = base64.b64encode(img_bytes).decode()
        return f'data:image/jpeg;base64,{b64}', 'SDXL'
    except Exception as e:
        last_err = f'SDXL: {e}'

    # ── 5. FLUX.1-dev — highest quality text→image ───────────────────────────
    try:
        img_bytes = _hf_inference_img(
            'black-forest-labs/FLUX.1-dev',
            GAME_PROMPT, NEG, width=576, height=832, steps=25
        )
        b64 = base64.b64encode(img_bytes).decode()
        return f'data:image/jpeg;base64,{b64}', 'FLUX.1-dev'
    except Exception as e:
        last_err = f'FLUX: {e}'

    raise RuntimeError(last_err or 'All 3D avatar models unavailable.')


def _generate_tpose_avatar_pil(person_path, upper_path=None, lower_path=None):
    """
    Generate a fashion-mannequin avatar with person's face and outfit overlaid.
    Returns PIL RGB image. Use _generate_tpose_avatar() for base64 output.
    """
    from PIL import Image, ImageDraw, ImageFilter

    W, H = 420, 660
    SKIN = (225, 180, 150)
    SHOE = (45, 35, 30)
    BG   = (240, 240, 244)

    canvas = Image.new('RGBA', (W, H), BG + (255,))
    draw   = ImageDraw.Draw(canvas)
    cx     = W // 2

    # ── Head ──────────────────────────────────────────────────────────────────
    hr = int(W * 0.095)
    hy = int(H * 0.108)
    draw.ellipse([cx-hr, hy-hr, cx+int(hr*0.95), hy+hr], fill=SKIN+(255,))

    # ── Neck ──────────────────────────────────────────────────────────────────
    nw = int(W * 0.048)
    neck_bot = int(H * 0.225)
    draw.rectangle([cx-nw, hy+hr-6, cx+nw, neck_bot], fill=SKIN+(255,))

    # ── Shoulders + Torso (tapered trapezoid) ─────────────────────────────────
    sh_w  = int(W * 0.215)   # shoulder half-width
    wst_w = int(W * 0.135)   # waist half-width
    torso_top = int(H * 0.220)
    torso_bot = int(H * 0.548)
    draw.polygon([
        (cx - sh_w,  torso_top),
        (cx + sh_w,  torso_top),
        (cx + wst_w, torso_bot),
        (cx - wst_w, torso_bot),
    ], fill=SKIN+(255,))

    # ── Arms hanging naturally at sides ───────────────────────────────────────
    arm_top_w = int(W * 0.065)
    arm_bot_w = int(W * 0.050)
    arm_top_y = int(H * 0.233)
    arm_bot_y = int(H * 0.552)
    # Left arm
    lax = cx - sh_w - 2
    draw.polygon([
        (lax,              arm_top_y),
        (lax - arm_top_w,  arm_top_y + int((arm_bot_y - arm_top_y)*0.05)),
        (lax - arm_bot_w,  arm_bot_y),
        (lax + 4,          arm_bot_y),
    ], fill=SKIN+(255,))
    # Right arm
    rax = cx + sh_w + 2
    draw.polygon([
        (rax,              arm_top_y),
        (rax + arm_top_w,  arm_top_y + int((arm_bot_y - arm_top_y)*0.05)),
        (rax + arm_bot_w,  arm_bot_y),
        (rax - 4,          arm_bot_y),
    ], fill=SKIN+(255,))

    # ── Hips (wider than waist) ────────────────────────────────────────────────
    hip_w   = int(W * 0.165)
    hip_top = torso_bot
    hip_bot = int(H * 0.590)
    draw.polygon([
        (cx - wst_w, hip_top),
        (cx + wst_w, hip_top),
        (cx + hip_w, hip_bot),
        (cx - hip_w, hip_bot),
    ], fill=SKIN+(255,))

    # ── Legs (slightly tapered) ────────────────────────────────────────────────
    lw   = int(W * 0.105)
    lgap = int(W * 0.020)
    lt   = int(H * 0.582)
    lb   = int(H * 0.935)
    taper = int(lw * 0.12)
    # Left leg
    draw.polygon([
        (cx - lgap - lw,        lt),
        (cx - lgap,             lt),
        (cx - lgap - taper,     lb),
        (cx - lgap - lw - taper,lb),
    ], fill=SKIN+(255,))
    # Right leg
    draw.polygon([
        (cx + lgap,             lt),
        (cx + lgap + lw,        lt),
        (cx + lgap + lw + taper,lb),
        (cx + lgap + taper,     lb),
    ], fill=SKIN+(255,))

    # ── Shoes ─────────────────────────────────────────────────────────────────
    draw.ellipse([cx-lgap-lw-taper-8, int(H*0.922), cx-lgap-taper+14,    int(H*0.960)], fill=SHOE+(255,))
    draw.ellipse([cx+lgap+taper-14,   int(H*0.922), cx+lgap+lw+taper+8,  int(H*0.960)], fill=SHOE+(255,))

    # slight shadow under each shoe
    draw.ellipse([cx-lgap-lw-taper-4, int(H*0.955), cx-lgap-taper+10,   int(H*0.968)], fill=(180,170,165,180))
    draw.ellipse([cx+lgap+taper-10,   int(H*0.955), cx+lgap+lw+taper+4, int(H*0.968)], fill=(180,170,165,180))

    # ── Upper garment (shirt/top) over torso+arms region ──────────────────────
    if upper_path and os.path.isfile(upper_path):
        try:
            shirt = Image.open(upper_path).convert('RGBA')
            shirt = _remove_background(shirt, tolerance=50)
            shirt = _autocrop_transparent(shirt, padding=5)
            gw, gh = shirt.size
            # Fill full shoulder width, preserve aspect ratio
            target_w = int(W * 0.88)
            target_h = int(gh * target_w / gw)
            # Cap height so it doesn't overflow torso area
            max_h = int(H * 0.37)
            if target_h > max_h:
                target_h = max_h
                target_w = int(gw * target_h / gh)
            shirt = shirt.resize((target_w, target_h), Image.LANCZOS)
            paste_x = (W - target_w) // 2
            paste_y = int(H * 0.215)
            canvas.paste(shirt, (paste_x, paste_y), shirt)
        except Exception:
            pass

    # ── Lower garment (pants/skirt) over hips+legs region ─────────────────────
    if lower_path and os.path.isfile(lower_path):
        try:
            pants = Image.open(lower_path).convert('RGBA')
            pants = _remove_background(pants, tolerance=50)
            pants = _autocrop_transparent(pants, padding=5)
            gw, gh = pants.size
            target_w = int(W * 0.62)
            target_h = int(gh * target_w / gw)
            max_h = int(H * 0.44)
            if target_h > max_h:
                target_h = max_h
                target_w = int(gw * target_h / gh)
            pants = pants.resize((target_w, target_h), Image.LANCZOS)
            paste_x = (W - target_w) // 2
            paste_y = int(H * 0.537)
            canvas.paste(pants, (paste_x, paste_y), pants)
        except Exception:
            pass

    # ── Face photo on head (circular crop from top-centre of photo) ───────────
    if person_path and os.path.isfile(person_path):
        try:
            face = Image.open(person_path).convert('RGBA')
            fw, fh = face.size
            # crop top-centre square (face is usually at top)
            crop_size = min(fw, int(fh * 0.45))
            left  = (fw - crop_size) // 2
            face  = face.crop((left, 0, left + crop_size, crop_size))
            fs    = (hr * 2) - 6
            face  = face.resize((fs, fs), Image.LANCZOS)
            mask  = Image.new('L', (fs, fs), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, fs, fs], fill=255)
            # soften mask edge
            mask  = mask.filter(ImageFilter.GaussianBlur(radius=1))
            face.putalpha(mask)
            canvas.paste(face, (cx - fs//2, hy - fs//2), face)
        except Exception:
            pass

    return canvas.convert('RGB')


def _generate_tpose_avatar(person_path, upper_path=None, lower_path=None):
    """Same as _generate_tpose_avatar_pil but returns base64 JPEG string."""
    result = _generate_tpose_avatar_pil(person_path, upper_path, lower_path)
    buf = io.BytesIO()
    result.save(buf, format='JPEG', quality=93)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/jpeg;base64,{b64}'


def _build_outfit_composite(person_path, upper_path=None, lower_path=None, size=(512, 680)):
    """
    Combine person photo + upper + lower garments into a single clean composite.
    Uses rembg for AI background removal on all layers (cached session).
    Returns PIL RGB image ready for display or TripoSR input.
    """
    from PIL import Image

    W, H = size
    BG = (232, 232, 236)
    canvas = Image.new('RGBA', (W, H), BG + (255,))

    # ── 1. Person layer (bottom) ───────────────────────────────────────────────
    try:
        person = Image.open(person_path).convert('RGB')
        person_rgba = _rembg_remove(person)          # clean cutout
        pw, ph = person_rgba.size
        # Scale to fill ~90% of canvas height, centred
        scale   = min((W * 0.88) / pw, (H * 0.90) / ph)
        person_rgba = person_rgba.resize(
            (int(pw * scale), int(ph * scale)), Image.LANCZOS
        )
        px = (W - person_rgba.width)  // 2
        py =  H - person_rgba.height       # pin to bottom
        canvas.paste(person_rgba, (px, py), person_rgba)
    except Exception:
        pass   # continue even without person silhouette

    # ── 2. Upper garment (shirt / top) ────────────────────────────────────────
    if upper_path and os.path.isfile(upper_path):
        try:
            shirt = Image.open(upper_path).convert('RGB')
            shirt = _rembg_remove(shirt)              # RGBA
            shirt = _autocrop_transparent(shirt, padding=5)
            gw, gh = shirt.size
            target_w = int(W * 0.74)
            target_h = int(gh * target_w / gw)
            if target_h > int(H * 0.40):
                target_h = int(H * 0.40)
                target_w = int(gw * target_h / gh)
            shirt = shirt.resize((target_w, target_h), Image.LANCZOS)
            canvas.paste(shirt, ((W - target_w) // 2, int(H * 0.15)), shirt)
        except Exception:
            pass

    # ── 3. Lower garment (pants / skirt) ──────────────────────────────────────
    if lower_path and os.path.isfile(lower_path):
        try:
            pants = Image.open(lower_path).convert('RGB')
            pants = _rembg_remove(pants)              # RGBA
            pants = _autocrop_transparent(pants, padding=5)
            gw, gh = pants.size
            target_w = int(W * 0.60)
            target_h = int(gh * target_w / gw)
            if target_h > int(H * 0.46):
                target_h = int(H * 0.46)
                target_w = int(gw * target_h / gh)
            pants = pants.resize((target_w, target_h), Image.LANCZOS)
            canvas.paste(pants, ((W - target_w) // 2, int(H * 0.53)), pants)
        except Exception:
            pass

    return canvas.convert('RGB')


def _triposr_render_front(composite_pil):
    """
    Run local TripoSR on composite PIL image.
    Returns PIL Image of front-facing 3D render.
    Optimised for CPU: single front view, cached rembg session.
    """
    import torch
    import numpy as np
    from PIL import Image
    from tsr.utils import remove_background, resize_foreground

    model, device = _get_triposr_model()

    # Use cached rembg session (avoid slow cold-start on every call)
    session = _get_rembg_session()

    # TripoSR preprocessing: remove bg → grey-fill → normalise to [0,1]
    img = remove_background(composite_pil.convert('RGB'), session)
    img = resize_foreground(img, 0.85)
    arr = np.array(img).astype(np.float32) / 255.0
    # Alpha-composite onto grey 0.5 background (TripoSR standard preprocessing)
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    img_pil = Image.fromarray((arr * 255.0).astype(np.uint8))

    with torch.no_grad():
        scene_codes = model([img_pil], device=device)

    # Render 1 front view at 256x256 (4× faster than 4 views on CPU: saves ~110s)
    render_views = model.render(
        scene_codes, n_views=1,
        height=256, width=256,
        return_type='pil',
    )
    return render_views[0][0]


def _triposr_avatar(person_path, upper_path=None, lower_path=None):
    """
    Full pipeline: composite → TripoSR render → base64.
    Returns (base64_str, 'TripoSR').
    """
    from PIL import Image

    composite = _build_outfit_composite(person_path, upper_path, lower_path)
    front_view = _triposr_render_front(composite)

    buf = io.BytesIO()
    front_view.save(buf, format='JPEG', quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/jpeg;base64,{b64}', 'TripoSR'


def _composite_avatar_b64(person_path, upper_path=None, lower_path=None):
    """
    Fast composite (no 3D render) — person + clothes via rembg. Returns base64.
    """
    composite = _build_outfit_composite(person_path, upper_path, lower_path)
    buf = io.BytesIO()
    composite.save(buf, format='JPEG', quality=93)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/jpeg;base64,{b64}'


def _pil_avatar3d_fallback(person_path, outfit_desc):
    return _generate_tpose_avatar(person_path)


@login_required
@require_POST
def avatar3d_view(request):
    """
    Generate a 3D photorealistic avatar of the user wearing recommended outfit.
    Tries HuggingFace AI models first, falls back to PIL blue-bg composite.
    """
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)

    outfit_desc = data.get('outfit_desc', 'stylish outfit')

    if not request.user.profile_photo:
        return JsonResponse({
            'ok': False,
            'error': 'Please upload your photo first (click "Add Photo" on the avatar).'
        }, status=400)

    person_path = request.user.profile_photo.path
    if not os.path.isfile(person_path):
        return JsonResponse({'ok': False, 'error': 'Profile photo missing. Please re-upload.'}, status=400)

    # Try AI models first
    try:
        img_data, model_name = _try_avatar3d_huggingface(person_path, outfit_desc)
        return JsonResponse({'ok': True, 'image': img_data, 'method': f'3D AI ({model_name})'})
    except Exception:
        pass

    # PIL fallback — blue background 3D-style
    try:
        img_data = _pil_avatar3d_fallback(person_path, outfit_desc)
        return JsonResponse({'ok': True, 'image': img_data, 'method': '3D Preview (Smart)'})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'Avatar generation failed: {str(e)[:150]}'}, status=500)


@login_required
def quickavatar_view(request):
    """
    Instant avatar on photo upload.
    Uses rembg composite (real person photo on clean bg) if rembg is ready,
    else falls back to PIL mannequin.
    """
    if not request.user.profile_photo:
        return JsonResponse({'ok': False, 'error': 'No photo.'}, status=400)

    person_path = request.user.profile_photo.path
    if not os.path.isfile(person_path):
        return JsonResponse({'ok': False, 'error': 'Photo file missing.'}, status=400)

    # rembg composite — real person photo with clean background
    if _REMBG_SESSION is not None:
        try:
            img_data = _composite_avatar_b64(person_path)
            return JsonResponse({'ok': True, 'image': img_data, 'method': 'AI Avatar'})
        except Exception:
            pass

    # PIL mannequin fallback (rembg not loaded yet or failed)
    try:
        img_data = _generate_tpose_avatar(person_path)
        return JsonResponse({'ok': True, 'image': img_data, 'method': 'Avatar'})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)[:150]}, status=500)


@login_required
@require_POST
def tposeavatar_view(request):
    """
    Instant avatar with outfit — uses PIL mannequin directly (no rembg, no network).
    Returns in < 1 second. Frontend then calls /api/triposr/ for 3D upgrade.
    """
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)

    if not request.user.profile_photo:
        return JsonResponse({'ok': False, 'error': 'No photo.'}, status=400)

    person_path = request.user.profile_photo.path
    if not os.path.isfile(person_path):
        return JsonResponse({'ok': False, 'error': 'Photo missing.'}, status=400)

    def resolve(url):
        if not url:
            return None
        rel = url.replace(settings.MEDIA_URL, '').lstrip('/')
        p   = os.path.join(settings.MEDIA_ROOT, rel)
        return p if os.path.isfile(p) else None

    upper_path = resolve(data.get('upper_url', ''))
    lower_path = resolve(data.get('lower_url', ''))

    # PIL composite tryon — person photo with clothes overlaid at correct body regions
    # This shows the actual person wearing the outfit (not floating clothes)
    try:
        img_data = _pil_composite_tryon(person_path, upper_path, lower_path)
        return JsonResponse({'ok': True, 'image': img_data, 'method': 'StyleSense Overlay'})
    except Exception:
        pass

    # PIL mannequin fallback
    try:
        img_data = _generate_tpose_avatar(person_path, upper_path, lower_path)
        return JsonResponse({'ok': True, 'image': img_data, 'method': 'Mannequin'})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)[:150]}, status=500)


@login_required
@require_POST
def triposr_view(request):
    """
    3D avatar with outfit — LOCAL TripoSR first, then progressive fallbacks.
    Pipeline:
      1. LOCAL TripoSR (cached weights, all deps installed — best 3D quality)
      2. HuggingFace AI Try-On (IDM-VTON etc. — photorealistic 2D)
      3. rembg composite (person silhouette + clothes, local AI)
      4. PIL mannequin (always works)
    """
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({'ok': False, 'error': 'Invalid request.'}, status=400)

    if not request.user.profile_photo:
        return JsonResponse({'ok': False, 'error': 'No photo.'}, status=400)

    person_path = request.user.profile_photo.path
    if not os.path.isfile(person_path):
        return JsonResponse({'ok': False, 'error': 'Photo missing.'}, status=400)

    def resolve(url):
        if not url:
            return None
        rel = url.replace(settings.MEDIA_URL, '').lstrip('/')
        p   = os.path.join(settings.MEDIA_ROOT, rel)
        return p if os.path.isfile(p) else None

    upper_path  = resolve(data.get('upper_url', ''))
    lower_path  = resolve(data.get('lower_url', ''))
    outfit_desc = data.get('outfit_desc', 'stylish outfit')

    # ── 1. LOCAL TripoSR — 3D mesh render from outfit composite ───────────────
    if os.path.isdir(_TRIPOSR_DIR):
        import threading
        _result = [None]
        _error  = [None]

        def _run():
            try:
                img_b64, label = _triposr_avatar(person_path, upper_path, lower_path)
                _result[0] = (img_b64, label)
            except Exception as exc:
                _error[0] = str(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=300)   # wait up to 5 min (CPU: ~170s first run, ~90s cached)

        if _result[0]:
            img_data, label = _result[0]
            return JsonResponse({'ok': True, 'image': img_data, 'method': f'TripoSR 3D Local'})

    # ── 2. HuggingFace AI Try-On (IDM-VTON — photorealistic) ─────────────────
    if upper_path:
        try:
            img_data, space = _try_huggingface(person_path, upper_path, outfit_desc)
            if lower_path:
                img_data = _apply_lower_on_b64(img_data, lower_path)
            return JsonResponse({'ok': True, 'image': img_data, 'method': f'AI Try-On ({space})'})
        except Exception:
            pass

    # ── 3. rembg composite (person photo + clothes, local AI) ─────────────────
    try:
        img_data = _composite_avatar_b64(person_path, upper_path, lower_path)
        return JsonResponse({'ok': True, 'image': img_data, 'method': 'AI Composite'})
    except Exception:
        pass

    # ── 4. PIL mannequin fallback ──────────────────────────────────────────────
    try:
        img_data = _generate_tpose_avatar(person_path, upper_path, lower_path)
        return JsonResponse({'ok': True, 'image': img_data, 'method': 'Mannequin'})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)[:150]}, status=500)
