# Generated by Django 4.2.3 on 2023-09-13 19:33

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("categories", "0008_alter_authorities_last_training_date"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="categories",
            unique_together={("name", "authority")},
        ),
    ]
