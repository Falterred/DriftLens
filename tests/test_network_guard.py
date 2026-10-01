"""Offline network guard (Step 6)."""

from __future__ import annotations

import socket

import pytest

from driftlens.security.network_guard import (
    NetworkBlockedError,
    guard_active,
    network_guard,
)


def test_connect_blocked() -> None:
    with network_guard():
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("127.0.0.1", 9))


def test_getaddrinfo_blocked() -> None:
    with network_guard():
        with pytest.raises(NetworkBlockedError):
            socket.getaddrinfo("example.com", 80)


def test_unix_socketpair_allowed() -> None:
    with network_guard():
        left, right = socket.socketpair()
    left.close()
    right.close()


def test_restore() -> None:
    original = socket.getaddrinfo
    with network_guard():
        assert guard_active() is True
        assert socket.getaddrinfo is not original
    assert guard_active() is False
    assert socket.getaddrinfo is original
