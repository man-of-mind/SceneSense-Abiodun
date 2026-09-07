#!/usr/bin/env python3
"""Offline transaction regressions for spawn_blocker_v5 network metadata."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import carla

import spawn_blocker_v5 as blocker
from network_degradation_profile_v1 import (
    NETWORK_PROFILE_BLUEPRINT_ID,
    NETWORK_PROFILE_MANIFEST_PREFIX,
    NETWORK_PROFILE_ROLE_PREFIX,
    NetworkProfileError,
    build_manifest_role,
    build_zone_role,
    normalize_zone,
)


class FakeBlueprint:
    """Mutable blueprint stand-in; the library returns a fresh one per spawn."""

    def __init__(self):
        self.attributes = {}

    @staticmethod
    def has_attribute(name):
        return name in {"role_name", "sensor_tick"}

    def set_attribute(self, name, value):
        self.attributes[str(name)] = str(value)


class FakeBlueprintLibrary:
    @staticmethod
    def find(type_id):
        if type_id != NETWORK_PROFILE_BLUEPRINT_ID:
            raise IndexError(type_id)
        return FakeBlueprint()


class FakeActor:
    def __init__(self, world, actor_id, role_name, transform):
        self._world = world
        self.id = int(actor_id)
        self.type_id = NETWORK_PROFILE_BLUEPRINT_ID
        self.attributes = {"role_name": str(role_name)}
        self._transform = transform
        self.is_alive = True
        self.is_listening = False

    def get_transform(self):
        return self._transform

    def destroy(self):
        if not self.is_alive:
            return True
        self.is_alive = False
        self._world.destroyed_ids.append(self.id)
        return True


class StaleSpawnHandle:
    """Spawn return handle that cannot destroy; get_actor() can re-resolve it."""

    def __init__(self, actor):
        self._actor = actor

    @property
    def id(self):
        return self._actor.id

    @property
    def attributes(self):
        return self._actor.attributes

    @property
    def is_alive(self):
        return self._actor.is_alive

    @property
    def is_listening(self):
        return False

    def get_transform(self):
        return self._actor.get_transform()

    def destroy(self):
        self._actor._world.stale_destroy_attempt_ids.append(self.id)
        return False


class FakeActorList(list):
    def filter(self, type_pattern):
        if type_pattern == NETWORK_PROFILE_BLUEPRINT_ID:
            return FakeActorList(
                actor for actor in self if actor.type_id == NETWORK_PROFILE_BLUEPRINT_ID
            )
        return FakeActorList()


class FakeWorld:
    """Small CARLA-world double with controllable actor-list propagation."""

    def __init__(self, manifest_visible_on_enumeration=1, stale_spawn_handles=False):
        self._blueprints = FakeBlueprintLibrary()
        self._actors = {}
        self._next_actor_id = 1
        self._manifest_visible_on_enumeration = int(
            manifest_visible_on_enumeration
        )
        self._stale_spawn_handles = bool(stale_spawn_handles)
        self.enumeration_count = 0
        self.wait_count = 0
        self.spawned_ids = []
        self.destroyed_ids = []
        self.stale_destroy_attempt_ids = []

    def get_blueprint_library(self):
        return self._blueprints

    def _add_actor(self, role_name, transform):
        actor = FakeActor(
            self,
            self._next_actor_id,
            role_name,
            transform,
        )
        self._actors[actor.id] = actor
        self._next_actor_id += 1
        return actor

    def add_existing_zone(self, token, index, x_coord, y_coord, radius):
        return self._add_actor(
            build_zone_role(token, index, radius),
            carla.Transform(
                carla.Location(x=float(x_coord), y=float(y_coord), z=0.0),
                carla.Rotation(),
            ),
        )

    def add_existing_manifest(self, token, zone_count, start_active_sensors=False):
        return self._add_actor(
            build_manifest_role(token, zone_count, start_active_sensors),
            carla.Transform(carla.Location(), carla.Rotation()),
        )

    def spawn_actor(self, blueprint, transform):
        actor = self._add_actor(blueprint.attributes["role_name"], transform)
        self.spawned_ids.append(actor.id)
        if self._stale_spawn_handles:
            return StaleSpawnHandle(actor)
        return actor

    def get_actor(self, actor_id):
        actor = self._actors.get(int(actor_id))
        if actor is None or not actor.is_alive:
            return None
        return actor

    def get_actors(self):
        self.enumeration_count += 1
        visible = []
        for actor in self._actors.values():
            if not actor.is_alive:
                continue
            role_name = actor.attributes.get("role_name", "")
            if (
                role_name.startswith(NETWORK_PROFILE_MANIFEST_PREFIX)
                and self.enumeration_count < self._manifest_visible_on_enumeration
            ):
                continue
            visible.append(actor)
        return FakeActorList(visible)

    def wait_for_tick(self, _timeout=None):
        # A publisher must not advance the clock; this only models an external tick.
        self.wait_count += 1
        return types.SimpleNamespace()

    def live_profile_actors(self):
        return tuple(
            actor
            for actor in self._actors.values()
            if actor.is_alive
            and actor.attributes.get("role_name", "").startswith(
                NETWORK_PROFILE_ROLE_PREFIX
            )
        )


class NoTickWorld(FakeWorld):
    """Synchronous-world stand-in with no external clock owner."""

    def __init__(self):
        super().__init__()
        self.tick_call_count = 0

    def wait_for_tick(self, _timeout=None):
        self.wait_count += 1
        raise RuntimeError("no external synchronous-world tick")

    def tick(self):
        self.tick_call_count += 1
        raise AssertionError("passive blocker must never call world.tick()")


class NetworkProfileTransactionTests(unittest.TestCase):
    def setUp(self):
        self.zones = (
            normalize_zone(1, 8.827, 62.216, 18.0),
            normalize_zone(2, 90.390, 43.290, 18.0),
        )

    def test_fake_world_baseline_publishes_immediately_visible_profile(self):
        """Prove the fake supports the existing successful transaction path."""
        world = FakeWorld()
        with mock.patch.object(blocker, "new_session_token", return_value="00000000"):
            publication = blocker.publish_network_degradation_profile(
                world,
                self.zones,
                start_active_sensors=False,
            )

        self.assertTrue(publication.owns_actors)
        self.assertEqual("00000000", publication.session_token)
        self.assertEqual(3, len(world.live_profile_actors()))

    def test_publish_tolerates_delayed_manifest_actor_list_visibility(self):
        # Enumeration 1 is the startup conflict scan. Enumeration 2 sees only
        # freshly spawned zones; enumeration 3 finally sees the commit marker.
        world = FakeWorld(manifest_visible_on_enumeration=3)
        with mock.patch.object(blocker, "new_session_token", return_value="11111111"), \
                mock.patch.object(blocker.time, "sleep", return_value=None):
            publication = blocker.publish_network_degradation_profile(
                world,
                self.zones,
                start_active_sensors=False,
            )

        self.assertTrue(publication.owns_actors)
        self.assertEqual("11111111", publication.session_token)
        self.assertEqual(3, len(world.spawned_ids))
        self.assertEqual([], world.destroyed_ids)
        self.assertGreaterEqual(world.enumeration_count, 3)
        self.assertGreaterEqual(world.wait_count, 1)

    def test_failed_publish_rollback_re_resolves_stale_spawn_handles(self):
        world = FakeWorld(stale_spawn_handles=True)
        with mock.patch.object(blocker, "new_session_token", return_value="22222222"), \
                mock.patch.object(
                    blocker,
                    "discover_network_degradation_profile",
                    side_effect=NetworkProfileError("forced verification failure"),
                ), \
                mock.patch.object(blocker.time, "sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "unable to publish"):
                blocker.publish_network_degradation_profile(
                    world,
                    self.zones,
                    start_active_sensors=False,
                )

        self.assertEqual(3, len(world.spawned_ids))
        self.assertEqual((), world.live_profile_actors())
        self.assertEqual(set(world.spawned_ids), set(world.destroyed_ids))
        # The manifest is the last spawn and must be invalidated before zones.
        self.assertEqual(world.spawned_ids[-1], world.destroyed_ids[0])
        # Rollback must never trust the stale spawn-return proxies.
        self.assertEqual([], world.stale_destroy_attempt_ids)

    def test_startup_removes_persistent_zone_only_residue_without_replace_flag(self):
        world = FakeWorld()
        stale_ids = [
            world.add_existing_zone("33333333", 1, 8.827, 62.216, 18.0).id,
            world.add_existing_zone("33333333", 2, 90.390, 43.290, 18.0).id,
        ]
        with mock.patch.object(blocker, "new_session_token", return_value="44444444"), \
                mock.patch.object(blocker.time, "sleep", return_value=None), \
                mock.patch.object(
                    blocker,
                    "NETWORK_PROFILE_STALE_MIN_STABLE_SECONDS",
                    0.0,
                ):
            publication = blocker.publish_network_degradation_profile(
                world,
                self.zones,
                start_active_sensors=False,
                replace_existing=False,
            )

        self.assertTrue(publication.owns_actors)
        self.assertTrue(set(stale_ids).issubset(set(world.destroyed_ids)))
        live_roles = {
            actor.attributes["role_name"] for actor in world.live_profile_actors()
        }
        self.assertEqual(3, len(live_roles))
        self.assertTrue(all("44444444" in role_name for role_name in live_roles))
        self.assertGreaterEqual(
            world.wait_count,
            blocker.NETWORK_PROFILE_STALE_CONFIRMATION_TICKS,
        )

    def test_startup_removes_stable_matching_single_zone_residue(self):
        world = FakeWorld()
        stale_id = world.add_existing_zone(
            "66666666",
            1,
            8.827,
            62.216,
            18.0,
        ).id

        with mock.patch.object(blocker, "new_session_token", return_value="77777777"), \
                mock.patch.object(
                    blocker,
                    "NETWORK_PROFILE_STALE_MIN_STABLE_SECONDS",
                    0.0,
                ):
            publication = blocker.publish_network_degradation_profile(
                world,
                self.zones,
                start_active_sensors=False,
                replace_existing=False,
            )

        self.assertTrue(publication.owns_actors)
        self.assertIn(stale_id, world.destroyed_ids)
        self.assertEqual(3, len(world.live_profile_actors()))

    def test_startup_never_deletes_profile_that_commits_during_grace_period(self):
        # Visibility on enumeration 6 used to outlive the three-tick grace and
        # expose a manifest only after its zones had already been deleted.
        world = FakeWorld(manifest_visible_on_enumeration=6)
        world.add_existing_zone("55555555", 1, 8.827, 62.216, 18.0)
        world.add_existing_zone("55555555", 2, 90.390, 43.290, 18.0)
        world.add_existing_manifest("55555555", 2)

        publication = blocker.publish_network_degradation_profile(
            world,
            self.zones,
            start_active_sensors=False,
        )

        self.assertFalse(publication.owns_actors)
        self.assertEqual("55555555", publication.session_token)
        self.assertEqual([], world.spawned_ids)
        self.assertEqual([], world.destroyed_ids)
        self.assertGreaterEqual(world.wait_count, 1)

    def test_elapsed_grace_without_external_snapshots_never_deletes_residue(self):
        world = NoTickWorld()
        stale_ids = [
            world.add_existing_zone("88888888", 1, 8.827, 62.216, 18.0).id,
            world.add_existing_zone("88888888", 2, 90.390, 43.290, 18.0).id,
        ]
        monotonic_value = [100.0]

        def advancing_monotonic():
            monotonic_value[0] += 1.0
            return monotonic_value[0]

        with mock.patch.object(
            blocker.time,
            "monotonic",
            side_effect=advancing_monotonic,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot be safely reused"):
                blocker.publish_network_degradation_profile(
                    world,
                    self.zones,
                    start_active_sensors=False,
                    replace_existing=False,
                )

        self.assertGreater(world.wait_count, 0)
        self.assertEqual(0, world.tick_call_count)
        self.assertEqual([], world.spawned_ids)
        self.assertEqual([], world.destroyed_ids)
        self.assertEqual(
            set(stale_ids),
            {actor.id for actor in world.live_profile_actors()},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
