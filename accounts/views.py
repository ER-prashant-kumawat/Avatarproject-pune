from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .models import User, UserCounter


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'accounts/login.html')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        password = request.POST.get('password', '')
        confirm  = request.POST.get('confirm', '')
        if len(password) < 4:
            messages.error(request, 'Password must be at least 4 characters.')
            return render(request, 'accounts/register.html')
        if password != confirm:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'accounts/register.html')
        username = UserCounter.next_username()
        User.objects.create(username=username, password=make_password(password))
        messages.success(request, f'Account created! Your username is <strong>{username}</strong> — save it before closing this page.')
        return render(request, 'accounts/register.html', {'new_username': username})
    return render(request, 'accounts/register.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
@require_POST
def upload_avatar(request):
    photo = request.FILES.get('photo')
    if not photo:
        return JsonResponse({'ok': False, 'error': 'No photo provided.'}, status=400)

    allowed = ('image/jpeg', 'image/png', 'image/webp', 'image/gif')
    if photo.content_type not in allowed:
        return JsonResponse({'ok': False, 'error': 'Only JPG, PNG or WEBP allowed.'}, status=400)

    if photo.size > 8 * 1024 * 1024:
        return JsonResponse({'ok': False, 'error': 'Photo must be under 8 MB.'}, status=400)

    import os
    user = request.user
    # Delete old photo file if exists
    if user.profile_photo:
        try:
            old_path = user.profile_photo.path
            if os.path.isfile(old_path):
                os.remove(old_path)
        except Exception:
            pass

    user.profile_photo = photo
    user.save()

    return JsonResponse({'ok': True, 'photo_url': user.profile_photo.url})
