# Generated by Django 4.2.3 on 2023-07-18 21:25

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("datasets", "0007_alter_datasets_university"),
    ]

    operations = [
        migrations.AddField(
            model_name="datasets_university",
            name="vid",
            field=models.CharField(max_length=30, null=True),
        ),
    ]
