from django.contrib import admin
from .models import Outfit

@admin.register(Outfit)
class OutfitAdmin(admin.ModelAdmin):
    list_display = ['name', 'wear_type', 'occasions', 'color', 'style', 'user', 'is_dummy']
    list_filter  = ['wear_type', 'is_dummy', 'color', 'style']
