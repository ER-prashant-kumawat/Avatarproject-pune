from django.apps import AppConfig
import threading


class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'

    def ready(self):
        # Pre-load rembg AI model in a background thread at server startup.
        # This way the first avatar request is fast instead of waiting 30-60 sec.
        def _preload():
            try:
                import rembg
                from . import views
                views._REMBG_SESSION = rembg.new_session()
            except Exception:
                pass

        t = threading.Thread(target=_preload, daemon=True)
        t.start()
