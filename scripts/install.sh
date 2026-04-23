#!/bin/sh
set -eu

UNAME_S="${MM_UNAME_S:-$(uname -s)}"
UNAME_M="${MM_UNAME_M:-$(uname -m)}"
INSTALL_DIR="${MM_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${MM_VERSION:-latest}"
BASE_URL="${MM_RELEASE_BASE_URL:-https://github.com/iamazhar/muscle-memory/releases}"
BASE_URL="${BASE_URL%/}"

case "${UNAME_S}:${UNAME_M}" in
  Darwin:arm64)
    ASSET="mm-darwin-arm64"
    ;;
  Linux:x86_64)
    ASSET="mm-linux-x86_64"
    ;;
  *)
    echo "Unsupported platform: ${UNAME_S} ${UNAME_M}" >&2
    exit 1
    ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install muscle-memory" >&2
  exit 1
fi

if command -v shasum >/dev/null 2>&1; then
  sha256() {
    shasum -a 256 "$1" | awk '{print $1}'
  }
elif command -v sha256sum >/dev/null 2>&1; then
  sha256() {
    sha256sum "$1" | awk '{print $1}'
  }
else
  echo "shasum or sha256sum is required to verify release checksums" >&2
  exit 1
fi

if [ "${VERSION}" = "latest" ]; then
  DOWNLOAD_ROOT="${BASE_URL}/latest/download"
else
  DOWNLOAD_ROOT="${BASE_URL}/download/v${VERSION}"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT HUP INT TERM

curl -fsSL "${DOWNLOAD_ROOT}/${ASSET}" -o "${TMP_DIR}/${ASSET}"
curl -fsSL "${DOWNLOAD_ROOT}/SHA256SUMS" -o "${TMP_DIR}/SHA256SUMS"

EXPECTED="$(awk -v asset="${ASSET}" '$2 == asset { print $1 }' "${TMP_DIR}/SHA256SUMS")"
if [ -z "${EXPECTED}" ]; then
  echo "Checksum manifest missing entry for ${ASSET}" >&2
  exit 1
fi

ACTUAL="$(sha256 "${TMP_DIR}/${ASSET}")"
if [ "${EXPECTED}" != "${ACTUAL}" ]; then
  echo "Checksum verification failed for ${ASSET}" >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}"
install -m 0755 "${TMP_DIR}/${ASSET}" "${INSTALL_DIR}/mm"

echo "Installed mm to ${INSTALL_DIR}/mm"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo "Add ${INSTALL_DIR} to PATH to run mm from a new shell"
    ;;
esac
