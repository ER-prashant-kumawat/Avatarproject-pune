from django.urls import path
from . import views

urlpatterns = [
    path('api/recommend/',    views.get_recommendations, name='recommend'),
    path('api/tryon/',        views.try_on_view,         name='tryon'),
    path('api/avatar3d/',     views.avatar3d_view,       name='avatar3d'),
    path('api/quickavatar/',  views.quickavatar_view,    name='quickavatar'),
    path('api/tposeavatar/',  views.tposeavatar_view,    name='tposeavatar'),
    path('api/triposr/',      views.triposr_view,        name='triposr'),
]
