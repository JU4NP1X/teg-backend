# Generated by Django 4.2.3 on 2023-09-08 16:16

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("categories", "0006_alter_translations_unique_together"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="translations",
            unique_together={("language", "category")},
        ),
    ]