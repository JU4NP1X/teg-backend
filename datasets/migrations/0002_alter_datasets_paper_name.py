# Generated by Django 4.2.3 on 2023-07-10 03:09

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("datasets", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="datasets",
            name="paper_name",
            field=models.CharField(max_length=200, unique=True),
        ),
    ]
