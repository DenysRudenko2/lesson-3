#!/usr/bin/env bash
# install_dev_tools.sh
# Idempotent installer of Docker, Docker Compose, Python (>=3.9), pip and ML libs
# (torch, torchvision, pillow, Django) for the MLOps lesson-3 environment.
#
# Usage:
#   chmod +x install_dev_tools.sh
#   ./install_dev_tools.sh
#
# All actions are appended to ./install.log alongside stdout/stderr.

set -Eeuo pipefail

LOG_FILE="${LOG_FILE:-$(pwd)/install.log}"
PY_MIN_MAJOR=3
PY_MIN_MINOR=9
PY_PACKAGES=(torch torchvision pillow Django)

# ----------------------------- logging helpers -------------------------------
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(ts)] WARN: $*" | tee -a "$LOG_FILE" >&2; }
die()  { echo "[$(ts)] ERROR: $*" | tee -a "$LOG_FILE" >&2; exit 1; }

trap 'warn "Script failed at line $LINENO (cmd: $BASH_COMMAND)"' ERR

# Mirror everything to install.log
exec > >(tee -a "$LOG_FILE") 2>&1

log "===== install_dev_tools.sh started ====="
log "Working dir: $(pwd)"
log "User: $(id -un)  Host: $(hostname)  OS: $(uname -srm)"

# ----------------------------- privilege helper ------------------------------
SUDO=""
if [[ $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    die "Need root or sudo to install system packages."
  fi
fi

# ------------------------- detect package manager ----------------------------
if ! command -v apt-get >/dev/null 2>&1; then
  die "This script supports Debian/Ubuntu (apt-get). Adapt for other distros."
fi

APT_UPDATED=0
apt_update_once() {
  if [[ $APT_UPDATED -eq 0 ]]; then
    log "Running apt-get update..."
    $SUDO apt-get update -y
    APT_UPDATED=1
  fi
}

apt_install() {
  apt_update_once
  log "apt-get install: $*"
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "$@"
}

# ------------------------------- Docker --------------------------------------
install_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "Docker already installed: $(docker --version)"
    return
  fi
  log "Installing Docker via official convenience script..."
  apt_install ca-certificates curl gnupg
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  $SUDO sh /tmp/get-docker.sh
  rm -f /tmp/get-docker.sh
  if getent group docker >/dev/null && [[ -n "${SUDO_USER:-}" ]]; then
    $SUDO usermod -aG docker "${SUDO_USER}" || true
  fi
  log "Docker installed: $(docker --version)"
}

# --------------------------- Docker Compose ----------------------------------
install_docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    log "Docker Compose plugin already installed: $(docker compose version | head -n1)"
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    log "Legacy docker-compose found: $(docker-compose --version)"
    return
  fi
  log "Installing docker-compose-plugin..."
  apt_install docker-compose-plugin || apt_install docker-compose
  if docker compose version >/dev/null 2>&1; then
    log "Docker Compose installed: $(docker compose version | head -n1)"
  else
    log "Docker Compose installed: $(docker-compose --version)"
  fi
}

# ------------------------------- Python --------------------------------------
python_ok() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<PY
import sys
sys.exit(0 if sys.version_info >= (${PY_MIN_MAJOR}, ${PY_MIN_MINOR}) else 1)
PY
}

install_python() {
  if python_ok; then
    log "Python OK: $(python3 --version)"
    return
  fi
  log "Installing Python >= ${PY_MIN_MAJOR}.${PY_MIN_MINOR}..."
  apt_install python3 python3-venv python3-dev
  python_ok || die "Python >= ${PY_MIN_MAJOR}.${PY_MIN_MINOR} not available after install."
  log "Python installed: $(python3 --version)"
}

install_pip() {
  if command -v pip3 >/dev/null 2>&1; then
    log "pip already installed: $(pip3 --version)"
    return
  fi
  log "Installing pip..."
  apt_install python3-pip
  log "pip installed: $(pip3 --version)"
}

# ------------------------- Python ML libraries -------------------------------
pip_pkg_installed() {
  python3 - "$1" <<'PY'
import importlib.metadata, sys
name = sys.argv[1]
try:
    importlib.metadata.version(name)
    sys.exit(0)
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)
PY
}

install_python_packages() {
  local missing=()
  for pkg in "${PY_PACKAGES[@]}"; do
    if pip_pkg_installed "$pkg"; then
      log "Python pkg present: $pkg ($(python3 -c "import importlib.metadata as m; print(m.version('$pkg'))"))"
    else
      missing+=("$pkg")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "All Python packages already installed."
    return
  fi
  log "Installing missing Python packages: ${missing[*]}"
  # CPU-only torch wheels are smaller and enough for inference.
  local pip_cmd=(pip3 install --no-cache-dir --upgrade)
  if printf '%s\n' "${missing[@]}" | grep -qE '^(torch|torchvision)$'; then
    "${pip_cmd[@]}" --extra-index-url https://download.pytorch.org/whl/cpu "${missing[@]}"
  else
    "${pip_cmd[@]}" "${missing[@]}"
  fi
}

# ------------------------------ verification ---------------------------------
verify_versions() {
  log "----- version check -----"
  command -v docker         >/dev/null 2>&1 && log "docker:         $(docker --version)"            || warn "docker missing"
  docker compose version    >/dev/null 2>&1 && log "docker compose: $(docker compose version | head -n1)" \
    || { command -v docker-compose >/dev/null 2>&1 && log "docker-compose: $(docker-compose --version)" || warn "docker compose missing"; }
  command -v python3        >/dev/null 2>&1 && log "python3:        $(python3 --version)"           || warn "python3 missing"
  command -v pip3           >/dev/null 2>&1 && log "pip3:           $(pip3 --version)"               || warn "pip3 missing"
  for pkg in "${PY_PACKAGES[@]}"; do
    if pip_pkg_installed "$pkg"; then
      log "py-$pkg: $(python3 -c "import importlib.metadata as m; print(m.version('$pkg'))")"
    else
      warn "py-$pkg: NOT INSTALLED"
    fi
  done
  log "----- end version check -----"
}

# --------------------------------- main --------------------------------------
install_docker
install_docker_compose
install_python
install_pip
install_python_packages
verify_versions

log "===== install_dev_tools.sh finished OK ====="
