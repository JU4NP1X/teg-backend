# Generated by Django 4.2.3 on 2023-07-10 13:55

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("thesaurus", "0005_thesaurus_searched_for_datasets"),
    ]

    operations = [
        migrations.AlterField(
            model_name="translations",
            name="name",
            field=models.CharField(max_length=200, unique=True),
        ),
    ]
