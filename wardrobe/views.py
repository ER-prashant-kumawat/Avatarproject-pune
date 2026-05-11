import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_http_methods
from .models import Outfit, OCCASIONS
from recommender.engine import extract_dominant_color


@login_required
def dashboard(request):
    outfits_qs = Outfit.objects.filter(user=request.user)
    outfits = [{
        'id':        o.id,
        'name':      o.name,
        'wear_type': o.wear_type,
        'occasions': o.occasions,
        'color':     o.color,
        'style':     o.style,
        'image_url': o.image.url if o.image else '',
    } for o in outfits_qs]
    upper_count = sum(1 for o in outfits if o['wear_type'] == 'upper')
    lower_count = sum(1 for o in outfits if o['wear_type'] == 'lower')
    profile_photo_url = request.user.profile_photo.url if request.user.profile_photo else ''
    return render(request, 'dashboard/main.html', {
        'outfits':           outfits,
        'occasions':         OCCASIONS,
        'username':          request.user.username,
        'upper_count':       upper_count,
        'lower_count':       lower_count,
        'profile_photo_url': profile_photo_url,
    })


@login_required
@require_POST
def upload_outfit(request):
    name      = request.POST.get('name', '').strip()
    wear_type = request.POST.get('wear_type', '')
    occasions = request.POST.getlist('occasions')
    color     = request.POST.get('color', 'neutral')
    style     = request.POST.get('style', 'casual')
    image     = request.FILES.get('image')

    errors = []
    if not name:
        errors.append('Outfit name is required.')
    if wear_type not in ('upper', 'lower'):
        errors.append('Select Upper or Lower wear.')
    if not occasions:
        errors.append('Select at least one occasion.')
    if not image:
        errors.append('Please upload an image.')

    if errors:
        return JsonResponse({'ok': False, 'error': ' '.join(errors)}, status=400)

    outfit = Outfit.objects.create(
        user=request.user,
        name=name,
        wear_type=wear_type,
        occasions=','.join(occasions),
        color=color,
        style=style,
        image=image,
    )

    # Extract dominant color from image (ML feature)
    try:
        rgb = extract_dominant_color(outfit.image.path)
        outfit.dominant_rgb = json.dumps(rgb)
        outfit.save()
    except Exception:
        pass

    return JsonResponse({
        'ok': True,
        'id':         outfit.id,
        'name':       outfit.name,
        'wear_type':  outfit.wear_type,
        'occasions':  outfit.occasions,
        'image_url':  outfit.image.url,
        'color':      outfit.color,
        'style':      outfit.style,
    })


@login_required
@require_http_methods(['DELETE'])
def delete_outfit(request, pk):
    outfit = get_object_or_404(Outfit, pk=pk, user=request.user)
    try:
        if outfit.image and os.path.isfile(outfit.image.path):
            os.remove(outfit.image.path)
    except Exception:
        pass
    outfit.delete()
    return JsonResponse({'ok': True})


@login_required
def my_outfits_json(request):
    outfits = Outfit.objects.filter(user=request.user)
    data = [{
        'id':        o.id,
        'name':      o.name,
        'wear_type': o.wear_type,
        'occasions': o.occasions,
        'color':     o.color,
        'style':     o.style,
        'image_url': o.image.url if o.image else '',
    } for o in outfits]
    return JsonResponse(data, safe=False)
