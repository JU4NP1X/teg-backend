# Generated by Django 4.2.3 on 2023-08-29 19:31

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0002_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="documents",
            name="img",
            field=models.BinaryField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name="documents",
            name="pdf",
            field=models.BinaryField(blank=True, default=None, null=True),
        ),
    ]
