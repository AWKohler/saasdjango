# Generated by Django 4.2.3 on 2023-10-06 03:29

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0007_alter_bot_initalised"),
    ]

    operations = [
        migrations.RenameField(
            model_name="bot",
            old_name="initalised",
            new_name="initialised",
        ),
    ]
