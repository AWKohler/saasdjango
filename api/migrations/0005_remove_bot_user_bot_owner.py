# Generated by Django 4.2.3 on 2023-09-22 22:42

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0004_alter_document_file"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="bot",
            name="user",
        ),
        migrations.AddField(
            model_name="bot",
            name="owner",
            field=models.CharField(default="owna", max_length=255),
            preserve_default=False,
        ),
    ]