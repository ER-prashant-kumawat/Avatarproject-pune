from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Custom user with auto-generated numeric username (user1, user2 …)."""
    REQUIRED_FIELDS = []
    profile_photo = models.ImageField(upload_to='avatars/', null=True, blank=True)

    class Meta:
        db_table = 'accounts_user'

    def __str__(self):
        return self.username


class UserCounter(models.Model):
    """Single-row table that hands out the next user number atomically."""
    counter = models.PositiveIntegerField(default=0)

    class Meta:
        db_table = 'accounts_usercounter'

    @classmethod
    def next_username(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        obj.counter += 1
        obj.save()
        return f"user{obj.counter}"
