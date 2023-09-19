# Generated by Django 4.2.3 on 2023-09-17 19:41

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("categories", "0009_alter_categories_unique_together"),
    ]

    operations = [
        migrations.AddField(
            model_name="authorities",
            name="disabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name="authorities",
            name="active",
            field=models.BooleanField(default=False),
        ),
    ]