# Generated by Django 4.2.3 on 2023-09-10 16:01

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("categories", "0007_alter_translations_unique_together"),
    ]

    operations = [
        migrations.AlterField(
            model_name="authorities",
            name="last_training_date",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
