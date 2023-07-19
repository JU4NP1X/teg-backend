# Generated by Django 4.2.3 on 2023-07-18 20:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("thesaurus_datasets", "0006_datasets_university"),
    ]

    operations = [
        migrations.AlterField(
            model_name="datasets",
            name="university",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                to="thesaurus_datasets.datasets_university",
            ),
        ),
    ]
