import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stylesense.settings')

# Vercel: app variable required by @vercel/python builder
application = get_wsgi_application()
app = application

# ── Vercel cold-start setup ───────────────────────────────────────────────────
# Runs once per Lambda instance (not on every request)
if os.environ.get('VERCEL'):
    import shutil
    from pathlib import Path

    _BASE = Path(__file__).resolve().parent.parent
    _done_flag = '/tmp/.stylesense_ready'

    if not os.path.exists(_done_flag):
        try:
            # 1. Copy dummy outfit images from deployment → /tmp/media
            src = _BASE / 'media'
            dst = Path('/tmp/media')
            if src.is_dir() and not dst.exists():
                shutil.copytree(str(src), str(dst))
            dst.mkdir(parents=True, exist_ok=True)
            (dst / 'uploads').mkdir(exist_ok=True)
            (dst / 'avatars').mkdir(exist_ok=True)

            # 2. Run migrations on fresh /tmp/db.sqlite3
            from django.core.management import call_command
            call_command('migrate', '--run-syncdb', verbosity=0)

            # 3. Load dummy outfit fixture (only if DB is empty)
            from wardrobe.models import Outfit
            if not Outfit.objects.filter(is_dummy=True).exists():
                call_command('loaddata', 'dummy_outfits', verbosity=0)

            # Mark setup as done for this Lambda instance
            open(_done_flag, 'w').close()
        except Exception as e:
            # Don't crash the app on setup failure — log and continue
            import sys
            print(f'[StyleSense] Vercel setup warning: {e}', file=sys.stderr)
