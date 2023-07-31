from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.models import User


def bot_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/<bot's name>/documents/<filename>
    # return f'{filename}'
    return f'{instance.bot.name}/documents/{filename}'


class Bot(models.Model):
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=255)
    use = models.TextField()

    def __str__(self):
        return self.name


class Document(models.Model):
    bot = models.ForeignKey(Bot, related_name='documents', on_delete=models.CASCADE)
    # file = models.FileField(upload_to='{bot.name}/documents/')
    file = models.FileField(upload_to=bot_directory_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

# hmm
# Function to change bot document save path dynamically
# @receiver(pre_save, sender=Document)
# def update_file_path(sender, instance, **kwargs):
#     instance.file.name = bot_directory_path(instance, instance.file.name)

# import os
#
# # Function to change bot document save path dynamically
# @receiver(pre_save, sender=Document)
# def update_file_path(sender, instance, **kwargs):
#     file_name = os.path.basename(instance.file.name)
#     instance.file.name = bot_directory_path(instance, file_name)
