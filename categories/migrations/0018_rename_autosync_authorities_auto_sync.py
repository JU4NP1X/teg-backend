# Generated by Django 4.2.3 on 2023-09-22 18:44

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("categories", "0017_authorities_autosync_alter_categories_tree_id"),
    ]

    operations = [
        migrations.RenameField(
            model_name="authorities",
            old_name="autoSync",
            new_name="auto_sync",
        ),
    ]