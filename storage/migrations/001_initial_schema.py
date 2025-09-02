"""Peewee migrations -- 001_initial_schema.py.

Some examples (model - class or model name)::

    > Model = migrator.orm['table_name']            # Return model in current state by name
    > Model = migrator.ModelClass                   # Return model in current state by name

    > migrator.sql(sql)                             # Run custom SQL
    > migrator.run(func, *args, **kwargs)           # Run python function with the given args
    > migrator.create_model(Model)                  # Create a model (could be used as decorator)
    > migrator.remove_model(model, cascade=True)    # Remove a model
    > migrator.add_fields(model, **fields)          # Add fields to a model
    > migrator.change_fields(model, **fields)       # Change fields
    > migrator.remove_fields(model, *field_names, cascade=True)
    > migrator.rename_field(model, old_field_name, new_field_name)
    > migrator.rename_table(model, new_table_name)
    > migrator.add_index(model, *col_names, unique=False)
    > migrator.add_not_null(model, *field_names)
    > migrator.add_default(model, field_name, default)
    > migrator.add_constraint(model, name, sql)
    > migrator.drop_index(model, *col_names)
    > migrator.drop_not_null(model, *field_names)
    > migrator.drop_constraints(model, *constraints)

"""

from contextlib import suppress
from datetime import datetime

import peewee as pw
from peewee_migrate import Migrator

with suppress(ImportError):
    import playhouse.postgres_ext as pw_pext


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    """Write your migrations here."""

    # We perform different migrations for SQLite and other databases
    # This is because SQLite is very loose with enforcing its schema, and trying to migrate other databases like SQLite
    # will require per-database SQL queries.
    # Instead, we assume that because external DB support was added at a later date, it is safe to assume a newer base
    # schema instead of trying to migrate from an older schema.
    migrate_external(migrator, database, fake=fake)


def migrate_external(migrator: Migrator, database: pw.Database, *, fake=False):
    @migrator.create_model
    class Project(pw.Model):
        project_id = pw.UUIDField(primary_key=True)
        name = pw.CharField(null=False)
        account_id = pw.UUIDField(null=False)
        sandbox = pw_pext.BinaryJSONField(default={})
        is_public = pw.BooleanField(default=False)
        created_at = pw.TimestampField(null=False, default=datetime.now())
        updated_at = pw.TimestampField(null=False, default=datetime.now())

        class Meta:
            table_name = "project"

    @migrator.create_model
    class Message(pw.Model):
        id = pw.UUIDField(unique=True)
        thread_id = pw.UUIDField(null=False)
        type = pw.CharField(null=False)
        is_llm_message = pw.BooleanField(null=False, default=True)
        content = pw_pext.BinaryJSONField(default={})
        metadata = pw_pext.BinaryJSONField(default={})
        created_at = pw.TimestampField(null=False, default=datetime.now())
        updated_at = pw.TimestampField(null=False, default=datetime.now())

        class Meta:
            table_name = "message"


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Write your rollback migrations here."""

    migrator.remove_model("message")

    migrator.remove_model("project")
