from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import shutil

from lautools.db import (
    LaupyDB,
    LaupyRecipeCollection,
    LaupyRecipeInstance,
    LautoolsLocation,
)


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


class CollectionManager:
    def __init__(self, db: LaupyDB):
        self.db = db

    def _collection_root(self, location_id: int) -> Path:
        location = self.db.get_location(location_id)
        if location is None:
            raise ValueError(f"Location {location_id} does not exist")
        if location.disk_location is None:
            raise ValueError(f"Location {location.name!r} has no disk location")
        root = Path(location.disk_location).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Collection root is not a directory: {root}")
        return root

    def get_collection_location(self, location_id: int) -> LautoolsLocation:
        location = self.db.get_location(location_id)
        if location is None:
            raise ValueError(f"Location {location_id} does not exist")
        return location

    def get_recipe_instance_path(
        self,
        recipe_instance: LaupyRecipeInstance | int,
    ) -> Path:
        instance = (
            self.db.get_recipe_instance(recipe_instance)
            if isinstance(recipe_instance, int)
            else recipe_instance
        )
        if instance is None or instance.id is None:
            raise ValueError("Recipe instance does not exist")
        root = self._collection_root(instance.collection_location_id)
        return (root / instance.relative_path).resolve()

    def sync_recipe_instances_from_disk(
        self,
        collection: LaupyRecipeCollection | int,
    ) -> list[LaupyRecipeInstance]:
        location_id = (
            collection.location_id
            if isinstance(collection, LaupyRecipeCollection)
            else collection
        )
        root = self._collection_root(location_id)
        existing = {
            str(instance.relative_path): instance
            for instance in self.db.list_recipe_instances_for_collection(
                location_id
            )
        }

        instances: list[LaupyRecipeInstance] = []
        for entry in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            relative = Path(entry.name)
            instance = existing.get(str(relative))
            if instance is None:
                instance = self.db.add_recipe_instance(
                    LaupyRecipeInstance(
                        id=None,
                        collection_location_id=location_id,
                        name=entry.name,
                        relative_path=relative,
                        created_at=_now(),
                        last_inspected=_now(),
                    )
                )
            else:
                instance = self.db.update_recipe_instance(
                    replace(instance, last_inspected=_now())
                )
            instances.append(instance)

        self.db.connection.commit()
        return instances

    def clone_recipe_instance(
        self,
        source_instance: LaupyRecipeInstance | int,
        destination_collection: LaupyRecipeCollection | int,
        new_name: str,
    ) -> LaupyRecipeInstance:
        source = (
            self.db.get_recipe_instance(source_instance)
            if isinstance(source_instance, int)
            else source_instance
        )
        if source is None or source.id is None:
            raise ValueError("Source recipe instance does not exist")

        destination_collection_id = (
            destination_collection.location_id
            if isinstance(destination_collection, LaupyRecipeCollection)
            else destination_collection
        )
        destination = self.db.get_recipe_collection(destination_collection_id)
        if destination is None:
            raise ValueError(
                f"Destination collection {destination_collection_id} does not exist"
            )
        if not destination.use_as_workbench:
            raise ValueError("Destination collection is not a workbench")

        source_path = self.get_recipe_instance_path(source)
        if not source_path.is_dir():
            raise ValueError(f"Source recipe instance is missing: {source_path}")

        destination_root = self._collection_root(destination_collection_id)
        destination_relative = Path(new_name.strip())
        if (
            not destination_relative.name
            or destination_relative != Path(destination_relative.name)
        ):
            raise ValueError("new_name must be a single directory name")

        destination_path = destination_root / destination_relative
        if os.path.lexists(destination_path):
            raise ValueError(
                f"Destination recipe instance already exists:\n{destination_path}"
            )

        shutil.copytree(
            source_path,
            destination_path,
            symlinks=True,
        )

        created = self.db.add_recipe_instance(
            LaupyRecipeInstance(
                id=None,
                collection_location_id=destination_collection_id,
                name=destination_relative.name,
                relative_path=destination_relative,
                source_instance_id=source.id,
                created_at=_now(),
                cloned_at=_now(),
                last_inspected=_now(),
            )
        )
        self.db.connection.commit()
        return created
