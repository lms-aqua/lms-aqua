"""Self-signed certificates for push mode.

Not decoration: ``navigator.mediaDevices.getUserMedia`` is only available in a
secure context, and ``http://192.168.x.y:8443`` is not one — the camera API is
simply absent there, with no prompt and no error worth reading. So the sender
page has to be served over TLS, and on a LAN there is no CA that will issue for
a private IP. A generated self-signed cert plus a one-time browser warning is
the honest way through.

Uses ``cryptography`` when installed and falls back to the ``openssl`` CLI, so
neither is a hard requirement.
"""

from __future__ import annotations

import datetime
import ipaddress
import os
import shutil
import ssl
import subprocess
from pathlib import Path

CERT_VALID_DAYS = 365


class TLSError(Exception):
    """No certificate could be generated with the tools available."""


def default_cert_dir() -> Path:
    """Somewhere per-user and stable, so the phone only warns once."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "lostcam"


def ensure_cert(
    cert_dir: Path | None = None,
    hosts: list[str] | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Return ``(cert_path, key_path)``, generating them if needed."""
    directory = Path(cert_dir) if cert_dir else default_cert_dir()
    directory.mkdir(parents=True, exist_ok=True)
    cert_path = directory / "lostcam-cert.pem"
    key_path = directory / "lostcam-key.pem"

    if not force and cert_path.exists() and key_path.exists():
        return cert_path, key_path

    hosts = hosts or ["localhost"]
    try:
        _generate_with_cryptography(cert_path, key_path, hosts)
    except ImportError:
        _generate_with_openssl(cert_path, key_path, hosts)
    return cert_path, key_path


def _generate_with_cryptography(cert_path: Path, key_path: Path, hosts: list[str]) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "LostCam"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LostCam"),
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERT_VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(_san_entries(hosts)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    _restrict(key_path)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _san_entries(hosts: list[str]) -> list:
    from cryptography import x509

    entries: list = []
    for host in hosts:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            entries.append(x509.DNSName(host))
    if not any(isinstance(e, x509.DNSName) and e.value == "localhost" for e in entries):
        entries.append(x509.DNSName("localhost"))
    return entries


def _generate_with_openssl(cert_path: Path, key_path: Path, hosts: list[str]) -> None:
    openssl = shutil.which("openssl")
    if not openssl:
        raise TLSError(
            "push mode needs a TLS certificate, but neither the 'cryptography' "
            "package nor the 'openssl' command is available.\n"
            "Install one of:\n"
            "  pip install cryptography\n"
            "  (or install the openssl CLI)\n"
            "Alternatively use pull mode with the iOS app, which needs no cert."
        )
    san = ",".join(
        (f"IP:{host}" if _is_ip(host) else f"DNS:{host}") for host in hosts
    )
    if "DNS:localhost" not in san:
        san += ",DNS:localhost"
    command = [
        openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", str(CERT_VALID_DAYS), "-subj", "/CN=LostCam/O=LostCam",
        "-addext", f"subjectAltName={san}",
    ]
    try:
        result = subprocess.run(  # noqa: S603 - argument list, never a shell
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TLSError(f"openssl failed: {exc}") from exc
    if result.returncode != 0:
        raise TLSError(f"openssl failed: {(result.stderr or '').strip()}")
    _restrict(key_path)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _restrict(path: Path) -> None:
    """Keep the private key owner-only where the OS supports it."""
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - Windows ACLs differ
        pass


def server_context(cert_path: Path, key_path: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context


def fingerprint(cert_path: Path) -> str:
    """SHA-256 fingerprint, so the warning the phone shows can be verified."""
    import hashlib

    pem = Path(cert_path).read_bytes()
    der = ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))
