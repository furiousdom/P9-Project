from django.db import models

# Create your models here.
class Drug(models.Model):
    primarykey = models.CharField(max_length=20, primary_key=True, null=False)
    name = models.TextField()
