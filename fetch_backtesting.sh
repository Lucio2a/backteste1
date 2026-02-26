#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-backtesting.py}"

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "Le dépôt existe déjà: $TARGET_DIR"
  exit 0
fi

URLS=(
  "https://github.com/kernc/backtesting.py"
  "https://ghproxy.com/https://github.com/kernc/backtesting.py"
  "https://gitclone.com/github.com/kernc/backtesting.py"
)

attempt_clone() {
  local url="$1"
  echo "\n==> Tentative: $url"
  if git clone --depth 1 "$url" "$TARGET_DIR"; then
    echo "\nTéléchargement réussi depuis: $url"
    return 0
  fi

  rm -rf "$TARGET_DIR"
  return 1
}

for url in "${URLS[@]}"; do
  if attempt_clone "$url"; then
    exit 0
  fi
done

echo
cat <<'MSG'
Échec du téléchargement depuis toutes les sources.

Causes probables dans cet environnement:
- Proxy sortant bloqué (CONNECT 403)
- Pas d'accès Internet sortant

À exécuter sur une machine avec Internet:
  git clone https://github.com/kernc/backtesting.py
MSG

exit 1
