#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Generate and maintain vehicle and pedestrian traffic in the simulation.

Version 1 preserves the stock CARLA traffic-generation behavior while adding
periodic, ownership-scoped replenishment for long-running demonstrations.
"""

import time

import carla

import argparse
import logging
from numpy import random


DEFAULT_TM_PORT = 8010
DEFAULT_REPLENISH_INTERVAL_S = 5.0
DEFAULT_POPULATION_LOG_INTERVAL_S = 60.0
VEHICLE_SPAWN_CLEARANCE_M = 4.0
WALKER_NAVIGATION_RETRIES = 30
WALKER_SPAWN_ROUNDS = 3
PERCENTAGE_PEDESTRIANS_RUNNING = 0.0
PERCENTAGE_PEDESTRIANS_CROSSING = 0.0


def positive_float(value):
    """Argparse converter for positive finite durations."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be a number')
    if parsed <= 0.0 or parsed == float('inf') or parsed != parsed:
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return parsed


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def actor_is_alive(actor):
    """Return False for missing, destroyed, or stale CARLA actor proxies."""
    if actor is None:
        return False
    try:
        return bool(actor.is_alive)
    except (AttributeError, RuntimeError):
        return False


class TrafficPopulationManager(object):
    """Own, validate, replenish, and clean up this client's traffic actors."""

    def __init__(
            self,
            client,
            world,
            traffic_manager,
            args,
            vehicle_blueprints,
            walker_blueprints,
            vehicle_spawn_points,
            synchronous_master):
        self.client = client
        self.world = world
        self.traffic_manager = traffic_manager
        self.vehicle_blueprints = vehicle_blueprints
        self.walker_blueprints = walker_blueprints
        self.vehicle_spawn_points = list(vehicle_spawn_points)
        self.target_vehicle_count = args.number_of_vehicles
        self.target_walker_count = args.number_of_walkers
        self.car_lights_on = args.car_lights_on
        self.hero_requested = args.hero
        self.synchronous_master = synchronous_master
        self.asynchronous = args.asynch
        self.tm_port = traffic_manager.get_port()
        self.replenish_interval = args.replenish_interval
        self.population_log_interval = args.population_log_interval
        self.vehicle_ids = []
        self.walkers = []
        self.orphan_controller_ids = set()
        self._last_population_log = 0.0
        self._walker_controller_blueprint = (
            world.get_blueprint_library().find('controller.ai.walker'))

    @property
    def _batch_ticks_world(self):
        return self.synchronous_master and not self.asynchronous

    def _apply_batch_sync(self, batch):
        if not batch:
            return []
        return self.client.apply_batch_sync(batch, self._batch_ticks_world)

    def _wait_for_actor_update(self):
        # An owner-side apply_batch_sync(..., True) already advances a frame.
        # Followers and asynchronous clients wait rather than stealing a tick.
        if not self._batch_ticks_world:
            self.world.wait_for_tick()

    def _live_actor_map(self, actor_ids):
        actor_ids = list(dict.fromkeys(
            actor_id for actor_id in actor_ids if actor_id is not None))
        if not actor_ids:
            return {}
        actors = self.world.get_actors(actor_ids)
        return {
            actor.id: actor
            for actor in actors
            if actor_is_alive(actor)
        }

    @staticmethod
    def _has_type(actor, prefix):
        return actor_is_alive(actor) and actor.type_id.startswith(prefix)

    def _random_navigation_location(self):
        for _ in range(WALKER_NAVIGATION_RETRIES):
            location = self.world.get_random_location_from_navigation()
            if location is not None:
                return location
        return None

    def _prepare_vehicle_blueprint(self, role_name):
        blueprint = random.choice(self.vehicle_blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', role_name)
        return blueprint

    def _tracked_hero_is_alive(self):
        live_vehicles = self._live_actor_map(self.vehicle_ids)
        for actor in live_vehicles.values():
            if not self._has_type(actor, 'vehicle.'):
                continue
            try:
                if actor.attributes.get('role_name') == 'hero':
                    return True
            except (AttributeError, RuntimeError):
                continue
        return False

    def _vehicle_spawn_point_is_clear(self, transform, occupied_locations):
        for location in occupied_locations:
            try:
                if transform.location.distance(location) < VEHICLE_SPAWN_CLEARANCE_M:
                    return False
            except (AttributeError, RuntimeError):
                continue
        return True

    def _spawn_vehicles(self, count, shuffle_spawn_points=True):
        if count <= 0:
            return []

        try:
            all_vehicle_actors = self.world.get_actors().filter('vehicle.*')
            occupied_locations = [
                actor.get_location()
                for actor in all_vehicle_actors
                if actor_is_alive(actor)
            ]
        except RuntimeError:
            # A failed world snapshot must not be interpreted as an empty map.
            raise

        spawn_points = list(self.vehicle_spawn_points)
        if shuffle_spawn_points:
            random.shuffle(spawn_points)

        hero_needed = self.hero_requested and not self._tracked_hero_is_alive()
        batch = []
        for transform in spawn_points:
            if len(batch) >= count:
                break
            if not self._vehicle_spawn_point_is_clear(
                    transform, occupied_locations):
                continue
            role_name = 'hero' if hero_needed else 'autopilot'
            hero_needed = False
            blueprint = self._prepare_vehicle_blueprint(role_name)
            batch.append(
                carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor,
                        True,
                        self.tm_port)))
            occupied_locations.append(transform.location)

        new_vehicle_ids = []
        for response in self._apply_batch_sync(batch):
            if response.error:
                logging.debug('Vehicle spawn failed: %s', response.error)
            else:
                new_vehicle_ids.append(response.actor_id)

        # Enroll successful IDs before making any additional fallible RPCs.
        self.vehicle_ids.extend(new_vehicle_ids)
        if self.car_lights_on and new_vehicle_ids:
            live_vehicles = self._live_actor_map(new_vehicle_ids)
            for actor in live_vehicles.values():
                try:
                    self.traffic_manager.update_vehicle_lights(actor, True)
                except RuntimeError as error:
                    logging.warning(
                        'Could not enable automatic lights for vehicle %d: %s',
                        actor.id,
                        error)
        return new_vehicle_ids

    @staticmethod
    def _walker_speed(walker_blueprint):
        if not walker_blueprint.has_attribute('speed'):
            logging.warning('Walker blueprint %s has no speed attribute',
                            walker_blueprint.id)
            return 0.0
        speed_values = walker_blueprint.get_attribute('speed').recommended_values
        speed_index = (
            1 if random.random() > PERCENTAGE_PEDESTRIANS_RUNNING else 2)
        if not speed_values:
            return 0.0
        speed_index = min(speed_index, len(speed_values) - 1)
        return float(speed_values[speed_index])

    def _spawn_walker_bodies_once(self, count):
        candidates = []
        batch = []
        for _ in range(count):
            location = self._random_navigation_location()
            if location is None:
                continue
            spawn_transform = carla.Transform()
            spawn_transform.location = location
            # Preserve the stock script's clearance above the navigation mesh.
            spawn_transform.location.z += 2.0
            walker_blueprint = random.choice(self.walker_blueprints)
            if walker_blueprint.has_attribute('is_invincible'):
                walker_blueprint.set_attribute('is_invincible', 'false')
            candidates.append({
                'id': None,
                'con': None,
                'speed': self._walker_speed(walker_blueprint),
                'controller_ready': False,
            })
            batch.append(carla.command.SpawnActor(
                walker_blueprint, spawn_transform))

        new_walkers = []
        for response, candidate in zip(self._apply_batch_sync(batch), candidates):
            if response.error:
                logging.debug('Walker spawn failed: %s', response.error)
            else:
                candidate['id'] = response.actor_id
                new_walkers.append(candidate)

        # Enroll bodies immediately so interruption cannot leak an unowned actor.
        self.walkers.extend(new_walkers)
        return new_walkers

    def _spawn_missing_walker_controllers(self, walker_records):
        walker_records = [
            record for record in walker_records
            if record.get('id') is not None and record.get('con') is None
        ]
        if not walker_records:
            return 0

        live_bodies = self._live_actor_map(
            [record['id'] for record in walker_records])
        candidates = [
            record for record in walker_records
            if self._has_type(live_bodies.get(record['id']),
                              'walker.pedestrian.')
        ]
        batch = [
            carla.command.SpawnActor(
                self._walker_controller_blueprint,
                carla.Transform(),
                record['id'])
            for record in candidates
        ]

        spawned = 0
        for response, record in zip(self._apply_batch_sync(batch), candidates):
            if response.error:
                logging.debug(
                    'Walker controller spawn failed for body %d: %s',
                    record['id'],
                    response.error)
            else:
                # Keep the ID even before initialization so cleanup owns it.
                record['con'] = response.actor_id
                record['controller_ready'] = False
                spawned += 1
        if spawned:
            self._wait_for_actor_update()
        return spawned

    def _initialize_walker_controllers(self, walker_records):
        candidates = [
            record for record in walker_records
            if record.get('con') is not None
            and not record.get('controller_ready', False)
        ]
        if not candidates:
            return 0

        actor_ids = []
        for record in candidates:
            actor_ids.extend((record['id'], record['con']))
        live_actors = self._live_actor_map(actor_ids)
        initialized = 0
        for record in candidates:
            body = live_actors.get(record['id'])
            controller = live_actors.get(record['con'])
            if not self._has_type(body, 'walker.pedestrian.'):
                continue
            if not self._has_type(controller, 'controller.ai.walker'):
                # Retain the returned controller ID as cleanup ownership until
                # a later authoritative snapshot confirms that it is absent.
                self.orphan_controller_ids.add(record['con'])
                record['con'] = None
                record['controller_ready'] = False
                continue
            try:
                parent = controller.parent
            except (AttributeError, RuntimeError):
                parent = None
            if parent is not None and parent.id != record['id']:
                self.orphan_controller_ids.add(record['con'])
                record['con'] = None
                record['controller_ready'] = False
                continue

            destination = self._random_navigation_location()
            if destination is None:
                continue
            try:
                controller.start()
                controller.go_to_location(destination)
                controller.set_max_speed(float(record['speed']))
            except RuntimeError as error:
                logging.debug(
                    'Walker controller %d initialization failed: %s',
                    record['con'],
                    error)
                continue
            record['controller_ready'] = True
            initialized += 1
        return initialized

    def _spawn_walkers(self, count):
        if count <= 0:
            return []
        new_walkers = []
        for _ in range(WALKER_SPAWN_ROUNDS):
            remaining = count - len(new_walkers)
            if remaining <= 0:
                break
            new_walkers.extend(self._spawn_walker_bodies_once(remaining))
        self._spawn_missing_walker_controllers(new_walkers)
        self._initialize_walker_controllers(new_walkers)
        return new_walkers

    def _stop_controller(self, controller):
        if not self._has_type(controller, 'controller.ai.walker'):
            return
        try:
            controller.stop()
        except RuntimeError:
            pass

    def _reap_orphan_controllers(self, live_actors):
        live_orphans = []
        for controller_id in list(self.orphan_controller_ids):
            actor = live_actors.get(controller_id)
            if not self._has_type(actor, 'controller.ai.walker'):
                self.orphan_controller_ids.discard(controller_id)
                continue
            self._stop_controller(actor)
            live_orphans.append(controller_id)
        if not live_orphans:
            return
        responses = self._apply_batch_sync([
            carla.command.DestroyActor(actor_id)
            for actor_id in live_orphans
        ])
        for controller_id, response in zip(live_orphans, responses):
            if response.error:
                logging.debug(
                    'Orphan walker controller %d cleanup failed: %s',
                    controller_id,
                    response.error)
            else:
                self.orphan_controller_ids.discard(controller_id)

    def _reconcile_owned_actors(self):
        actor_ids = list(self.vehicle_ids)
        for record in self.walkers:
            actor_ids.append(record.get('id'))
            actor_ids.append(record.get('con'))
        actor_ids.extend(self.orphan_controller_ids)

        # This single snapshot happens before any registry mutation. If the RPC
        # fails, the exception propagates and no replacement storm is possible.
        live_actors = self._live_actor_map(actor_ids)

        previous_vehicle_ids = list(self.vehicle_ids)
        self.vehicle_ids = [
            actor_id for actor_id in previous_vehicle_ids
            if self._has_type(live_actors.get(actor_id), 'vehicle.')
        ]
        lost_vehicle_count = len(previous_vehicle_ids) - len(self.vehicle_ids)

        retained_walkers = []
        lost_walker_count = 0
        for record in self.walkers:
            body_id = record.get('id')
            controller_id = record.get('con')
            body = live_actors.get(body_id)
            controller = live_actors.get(controller_id)
            if not self._has_type(body, 'walker.pedestrian.'):
                lost_walker_count += 1
                if self._has_type(controller, 'controller.ai.walker'):
                    self.orphan_controller_ids.add(controller_id)
                continue
            if controller_id is not None and not self._has_type(
                    controller, 'controller.ai.walker'):
                self.orphan_controller_ids.add(controller_id)
                record['con'] = None
                record['controller_ready'] = False
            elif controller_id is not None:
                try:
                    parent = controller.parent
                except (AttributeError, RuntimeError):
                    parent = None
                if parent is not None and parent.id != body_id:
                    self.orphan_controller_ids.add(controller_id)
                    record['con'] = None
                    record['controller_ready'] = False
            retained_walkers.append(record)
        self.walkers = retained_walkers

        self._reap_orphan_controllers(live_actors)
        return lost_vehicle_count, lost_walker_count

    def spawn_initial_population(self):
        self._spawn_vehicles(
            self.target_vehicle_count,
            shuffle_spawn_points=False)
        self._spawn_walkers(self.target_walker_count)
        self.log_population(force=True)

    def reconcile(self):
        lost_vehicle_count, lost_walker_count = self._reconcile_owned_actors()
        if lost_vehicle_count or lost_walker_count:
            logging.warning(
                'Detected population loss: vehicles=%d walkers=%d',
                lost_vehicle_count,
                lost_walker_count)

        # Repair controllers for surviving bodies before filling body deficits.
        self._spawn_missing_walker_controllers(self.walkers)
        self._initialize_walker_controllers(self.walkers)

        missing_vehicles = max(
            0, self.target_vehicle_count - len(self.vehicle_ids))
        new_vehicle_ids = self._spawn_vehicles(missing_vehicles)

        missing_walkers = max(
            0, self.target_walker_count - len(self.walkers))
        new_walkers = self._spawn_walkers(missing_walkers)

        if new_vehicle_ids or new_walkers:
            logging.info(
                'Replenished vehicles=%d walkers=%d',
                len(new_vehicle_ids),
                len(new_walkers))
        self.log_population()

    def log_population(self, force=False):
        now = time.monotonic()
        if not force and (
                now - self._last_population_log
                < self.population_log_interval):
            return
        self._last_population_log = now
        ready_controllers = sum(
            1 for record in self.walkers
            if record.get('controller_ready', False))
        logging.info(
            'Managed population: vehicles=%d/%d walkers=%d/%d '
            'active_controllers=%d/%d',
            len(self.vehicle_ids),
            self.target_vehicle_count,
            len(self.walkers),
            self.target_walker_count,
            ready_controllers,
            len(self.walkers))

    def destroy(self):
        """Best-effort cleanup of actors created by this script only."""
        controller_ids = set(self.orphan_controller_ids)
        body_ids = []
        for record in self.walkers:
            body_id = record.get('id')
            controller_id = record.get('con')
            if body_id is not None:
                body_ids.append(body_id)
            if controller_id is not None:
                controller_ids.add(controller_id)

        try:
            live_controllers = self._live_actor_map(controller_ids)
            for controller in live_controllers.values():
                self._stop_controller(controller)
        except RuntimeError as error:
            logging.warning('Could not stop all walker controllers: %s', error)

        cleanup_groups = (
            ('walker controllers', list(controller_ids)),
            ('walkers', body_ids),
            ('vehicles', list(self.vehicle_ids)),
        )
        for label, actor_ids in cleanup_groups:
            if not actor_ids:
                continue
            logging.info('Destroying %d managed %s', len(actor_ids), label)
            try:
                responses = self._apply_batch_sync([
                    carla.command.DestroyActor(actor_id)
                    for actor_id in actor_ids
                ])
                failures = sum(1 for response in responses if response.error)
                if failures:
                    logging.warning(
                        '%d managed %s could not be destroyed',
                        failures,
                        label)
            except RuntimeError as error:
                logging.warning('Could not destroy managed %s: %s', label, error)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles', metavar='N', default=30, type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers', metavar='W', default=10, type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe', action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv', metavar='PATTERN', default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv', metavar='G', default='All',
        help='restrict to certain vehicle generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--filterw', metavar='PATTERN', default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw', metavar='G', default='All',
        help='restrict to certain pedestrian generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--tm-port', metavar='P', default=DEFAULT_TM_PORT, type=int,
        help='Port to communicate with TM (default: %(default)s)')
    argparser.add_argument(
        '--asynch', action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid', action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed', metavar='S', type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw', metavar='S', default=0, type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on', action='store_true', default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero', action='store_true', default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn', action='store_true', default=False,
        help=(
            'Enable TM dormant-vehicle respawning on large maps; managed '
            'actor-count replenishment is always active'))
    argparser.add_argument(
        '--no-rendering', action='store_true', default=False,
        help='Activate no rendering mode')
    argparser.add_argument(
        '--replenish-interval', metavar='SECONDS',
        default=DEFAULT_REPLENISH_INTERVAL_S, type=positive_float,
        help=(
            'Seconds between managed actor health checks and replacement '
            'attempts (default: %(default)s)'))
    argparser.add_argument(
        '--population-log-interval', metavar='SECONDS',
        default=DEFAULT_POPULATION_LOG_INTERVAL_S, type=positive_float,
        help=(
            'Seconds between managed vehicle/walker count reports '
            '(default: %(default)s)'))

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    world = None
    traffic_manager = None
    population = None
    original_settings = None
    synchronous_settings_changed = False
    no_rendering_changed = False
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        logging.info(
            'Using Traffic Manager port %d; population check interval %.1fs',
            args.tm_port,
            args.replenish_interval)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        original_settings = world.get_settings()
        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                synchronous_settings_changed = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode, and traffic might experience some issues")

        if args.no_rendering and not settings.no_rendering_mode:
            no_rendering_changed = True
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if args.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
            if not blueprints:
                raise ValueError("Couldn't find any safe vehicles with the specified filters")

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        world.set_pedestrians_cross_factor(PERCENTAGE_PEDESTRIANS_CROSSING)

        population = TrafficPopulationManager(
            client,
            world,
            traffic_manager,
            args,
            blueprints,
            blueprintsWalkers,
            spawn_points,
            synchronous_master)
        population.spawn_initial_population()

        print(
            'spawned %d vehicles and %d walkers; maintaining targets %d/%d, '
            'press Ctrl+C to exit.' % (
                len(population.vehicle_ids),
                len(population.walkers),
                population.target_vehicle_count,
                population.target_walker_count))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        next_replenishment_at = (
            time.monotonic() + population.replenish_interval)
        while True:
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

            now = time.monotonic()
            if now < next_replenishment_at:
                continue
            next_replenishment_at = now + population.replenish_interval
            try:
                population.reconcile()
            except RuntimeError as error:
                # Preserve ownership registries and retry after the next
                # bounded interval instead of treating an RPC failure as loss.
                logging.error('Population maintenance failed: %s', error)

    finally:

        if population is not None:
            try:
                population.destroy()
            except Exception as error:
                logging.exception(
                    'Managed actor cleanup encountered an error: %s', error)

        if traffic_manager is not None and synchronous_master:
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError as error:
                logging.warning(
                    'Could not disable Traffic Manager synchronous mode: %s',
                    error)

        if world is not None and original_settings is not None and (
                synchronous_settings_changed or no_rendering_changed):
            try:
                settings = world.get_settings()
                if synchronous_settings_changed:
                    settings.synchronous_mode = original_settings.synchronous_mode
                    settings.fixed_delta_seconds = original_settings.fixed_delta_seconds
                if no_rendering_changed:
                    settings.no_rendering_mode = original_settings.no_rendering_mode
                world.apply_settings(settings)
            except RuntimeError as error:
                logging.warning('Could not restore world settings: %s', error)

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

