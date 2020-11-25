from django.db import models

# Create your models here.

class DrugXml(models.Model):
    name = models.TextField()
    content = models.TextField()
