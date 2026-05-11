from django.db import models
from accounts.models import User

OCCASIONS = [
    ('casual',   'Casual'),
    ('wedding',  'Wedding'),
    ('party',    'Party'),
    ('formal',   'Formal'),
    ('festive',  'Festive'),
    ('outdoor',  'Outdoor'),
    ('sports',   'Sports'),
]

WEAR_TYPES = [
    ('upper', 'Upper Wear'),
    ('lower', 'Lower Wear'),
]

COLORS = [
    ('white',   'White'),
    ('black',   'Black'),
    ('blue',    'Blue'),
    ('navy',    'Navy Blue'),
    ('red',     'Red'),
    ('pink',    'Pink'),
    ('green',   'Green'),
    ('gold',    'Gold / Yellow'),
    ('grey',    'Grey'),
    ('beige',   'Beige / Cream'),
    ('purple',  'Purple'),
    ('maroon',  'Maroon'),
    ('orange',  'Orange'),
    ('floral',  'Multi / Floral'),
    ('neutral', 'Neutral'),
]

STYLES = [
    ('casual',       'Casual'),
    ('formal',       'Formal'),
    ('traditional',  'Traditional / Ethnic'),
    ('elegant',      'Elegant'),
    ('sports',       'Sports'),
    ('fusion',       'Fusion'),
]


class Outfit(models.Model):
    user       = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    name       = models.CharField(max_length=120)
    wear_type  = models.CharField(max_length=10, choices=WEAR_TYPES)
    occasions  = models.CharField(max_length=200)   # comma-separated
    color      = models.CharField(max_length=20, choices=COLORS, default='neutral')
    style      = models.CharField(max_length=20, choices=STYLES, default='casual')
    image      = models.ImageField(upload_to='uploads/')
    is_dummy   = models.BooleanField(default=False)
    # ML-extracted dominant colors stored as JSON string  "[r,g,b]"
    dominant_rgb = models.CharField(max_length=50, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.wear_type})"

    def occasions_list(self):
        return [o.strip() for o in self.occasions.split(',') if o.strip()]

    def occasion_vector(self):
        """7-dim multi-hot vector over OCCASIONS list."""
        keys = [o[0] for o in OCCASIONS]
        return [1 if k in self.occasions_list() else 0 for k in keys]
