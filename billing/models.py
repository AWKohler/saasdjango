from django.db import models


class APIKeyUsage(models.Model):
    """Tracks usage of an API Key."""
    api_key_name = models.CharField(max_length=100, unique=True)
    count = models.IntegerField(default=0)

    def increment(self):
        """Increment the usage count."""
        self.count += 1
        self.save()

    def __str__(self):
        return self.api_key_name
