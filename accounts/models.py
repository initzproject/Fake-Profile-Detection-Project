from django.contrib.auth.models import User
from django.db import models

class ProfileData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    followers = models.IntegerField()
    following = models.IntegerField()
    bio = models.TextField()
    has_profile_photo = models.BooleanField()
    is_private = models.BooleanField()

