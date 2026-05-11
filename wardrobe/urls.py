from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/',              views.dashboard,       name='dashboard'),
    path('api/upload/',             views.upload_outfit,   name='upload_outfit'),
    path('api/delete/<int:pk>/',    views.delete_outfit,   name='delete_outfit'),
    path('api/my-outfits/',         views.my_outfits_json, name='my_outfits'),
]
