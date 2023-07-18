# Generated by Django 4.2.3 on 2023-07-18 12:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("datasets", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="DatasetsEnglishTranslations",
            fields=[
                (
                    "dataset",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        primary_key=True,
                        serialize=False,
                        to="datasets.datasets",
                    ),
                ),
                ("paper_name", models.CharField(max_length=500)),
                ("summary", models.TextField()),
            ],
        ),
    ]
