"""Process-wide offline guarantee: block every outbound network operation."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_BLOCK_MESSAGE = "outbound network access is blocked (offline guarantee)"

_AF_UNIX = getattr(socket, "AF_UNIX", None)
# The stdlib socketpair() fallback used on platforms without AF_UNIX opens a
# loopback TCP connection; loopback never leaves the host, so allow it there.
_ALLOW_LOOPBACK_FALLBACK = _AF_UNIX is None


def _is_loopback_address(address: Any) -> bool:
    if isinstance(address, tuple) and address:
        host = address[0]
    elif isinstance(address, str):
        host = address
    else:
        return False
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class NetworkBlockedError(RuntimeError):
    """Raised when code attempts an outbound network operation."""


_ORIGINALS: dict[str, Any] = {}
_ACTIVE = False


def _blocked(*args: object, **kwargs: object) -> Any:
    raise NetworkBlockedError(_BLOCK_MESSAGE)


def _local_only(self: socket.socket, address: Any) -> bool:
    if _AF_UNIX is not None and self.family == _AF_UNIX:
        return True
    return _ALLOW_LOOPBACK_FALLBACK and _is_loopback_address(address)


def _blocked_connect(self: socket.socket, address: Any) -> Any:
    if _local_only(self, address):
        return _ORIGINALS["connect"](self, address)
    raise NetworkBlockedError(_BLOCK_MESSAGE)


def _blocked_connect_ex(self: socket.socket, address: Any) -> Any:
    if _local_only(self, address):
        return _ORIGINALS["connect_ex"](self, address)
    raise NetworkBlockedError(_BLOCK_MESSAGE)


def _blocked_sendto(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
    if _AF_UNIX is not None and self.family == _AF_UNIX:
        return _ORIGINALS["sendto"](self, *args, **kwargs)
    raise NetworkBlockedError(_BLOCK_MESSAGE)


def install_network_guard() -> None:
    """Install the process-wide guard. A second call is a no-op."""
    global _ACTIVE
    if _ACTIVE:
        return
    if not _ORIGINALS:
        _ORIGINALS["connect"] = socket.socket.connect
        _ORIGINALS["connect_ex"] = socket.socket.connect_ex
        _ORIGINALS["sendto"] = socket.socket.sendto
        _ORIGINALS["create_connection"] = socket.create_connection
        _ORIGINALS["getaddrinfo"] = socket.getaddrinfo
        _ORIGINALS["gethostbyname"] = socket.gethostbyname
        _ORIGINALS["gethostbyname_ex"] = socket.gethostbyname_ex

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = _blocked_sendto  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]
    socket.gethostbyname_ex = _blocked  # type: ignore[assignment]
    _ACTIVE = True


def uninstall_network_guard() -> None:
    """Restore the original socket functions (tests only)."""
    global _ACTIVE
    if not _ACTIVE:
        return
    socket.socket.connect = _ORIGINALS["connect"]  # type: ignore[method-assign]
    socket.socket.connect_ex = _ORIGINALS["connect_ex"]  # type: ignore[method-assign]
    socket.socket.sendto = _ORIGINALS["sendto"]  # type: ignore[method-assign]
    socket.create_connection = _ORIGINALS["create_connection"]  # type: ignore[assignment]
    socket.getaddrinfo = _ORIGINALS["getaddrinfo"]  # type: ignore[assignment]
    socket.gethostbyname = _ORIGINALS["gethostbyname"]  # type: ignore[assignment]
    socket.gethostbyname_ex = _ORIGINALS["gethostbyname_ex"]  # type: ignore[assignment]
    _ACTIVE = False


def guard_active() -> bool:
    """Whether the outbound network guard is currently installed."""
    return _ACTIVE


@contextmanager
def network_guard() -> Iterator[None]:
    """Block outbound network access for the duration of the context."""
    was_active = _ACTIVE
    if not was_active:
        install_network_guard()
    try:
        yield
    finally:
        if not was_active:
            uninstall_network_guard()
