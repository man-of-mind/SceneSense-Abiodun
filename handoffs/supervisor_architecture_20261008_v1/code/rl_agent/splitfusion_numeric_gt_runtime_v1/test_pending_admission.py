"""Bounded pending rewards do not consume or bypass durable producer slots."""
import threading
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from . import ue_live as U
from .test_ue_live import handoff, planned, BOOT


class PendingAdmissionTests(unittest.TestCase):
    def bridge(self):
        bridge, first = handoff()
        bridge.registration['numeric_limits']['pending_reward_jobs'] = 2
        tickets = [first]
        for job in (2, 3, 4, 5):
            ticket = U._Ticket(job, 11 + job, {}, {},
                U.event('collector', 'owned_tick_snapshot', 100, 101, BOOT),
                source_owned=U.event('ue', 'source_owned', 110, 101, BOOT))
            bridge._tickets[ticket.frame_id] = ticket
            tickets.append(ticket)
        return bridge, tickets

    def test_third_and_fourth_plan_do_not_wait_or_reserve_producer_slot(self):
        bridge, tickets = self.bridge()
        for ticket in tickets[:2]:
            ticket.plan_bytes = b'already sent'
            ticket.producer_admitted = True
        for ticket in tickets[2:4]:
            plan = planned()
            plan.frame_id = plan.plan.frame_id = ticket.frame_id
            runtime = NS(engine=NS(lineage='a'*64), _numeric_current_ego_pose=(),
                pipeline=NS(stream_id='stream', decisions=[dict(frame_id=ticket.frame_id,
                    action_open=dict(domain='CLOCK_MONOTONIC_RAW', ns=120))]))
            with mock.patch.object(U, 'full_plan_bytes', return_value=(b'plan', {}, {}, {})):
                bridge.on_plan(runtime, plan)
            self.assertFalse(ticket.producer_admitted)
            self.assertEqual(ticket.action_open['raw_ns'], 120)
        self.assertEqual(bridge.client.calls, [])
        self.assertEqual(sum(t.producer_admitted for t in tickets), 2)
        fifth = tickets[4]
        plan.frame_id = plan.plan.frame_id = fifth.frame_id
        runtime.pipeline.decisions[-1]['frame_id'] = fifth.frame_id
        with mock.patch.object(U, 'full_plan_bytes', return_value=(b'plan', {}, {}, {})):
            with self.assertRaisesRegex(U.NumericUeError, 'capacity exceeded'):
                bridge.on_plan(runtime, plan)

    def test_pending_fifo_waits_for_actual_retirement_not_feedback_or_archive_notice(self):
        bridge, tickets = self.bridge()
        first, second, third, fourth, _ = tickets
        first.producer_admitted = second.producer_admitted = True
        bridge._producer_order.extend((third.job, fourth.job))
        entered, done, errors = threading.Event(), threading.Event(), []
        def wait():
            entered.set()
            try:
                bridge._await_prepare_turn(third)
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()
        thread = threading.Thread(target=wait)
        thread.start()
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(done.wait(.03))
            self.assertFalse(third.producer_admitted)
            # Feedback/timeout is not a capacity-release proof.
            first.terminal = U.event('ue', 'feedback_timeout', 200, 101, BOOT)
            with bridge._producer_turn:
                bridge._producer_turn.notify_all()
            self.assertFalse(done.wait(.03))
            with bridge._producer_turn:
                first.retired = True
                bridge._producer_turn.notify_all()
            self.assertTrue(done.wait(1))
            self.assertTrue(third.producer_admitted)
            self.assertEqual(sum(t.producer_admitted and not t.retired for t in tickets), 2)
            self.assertFalse(fourth.producer_admitted)
            self.assertEqual(errors, [])
        finally:
            with bridge._producer_turn:
                bridge._aborted = True
                bridge._producer_turn.notify_all()
            thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_pending_bound_is_registered_and_cannot_be_unlimited(self):
        bridge, _ = self.bridge()
        self.assertEqual(bridge._outstanding_limit(), 4)
        for value in (-1, 3, True, 2.0):
            bridge.registration['numeric_limits']['pending_reward_jobs'] = value
            with self.assertRaises(U.NumericUeError):
                bridge._outstanding_limit()


if __name__ == '__main__':
    unittest.main()
