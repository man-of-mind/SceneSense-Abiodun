"""Monitored control: identical startup collection, no freeze."""
from .client import Client as CandidateClient
from . import control_contract as C


class Client(CandidateClient):
    contract = C
