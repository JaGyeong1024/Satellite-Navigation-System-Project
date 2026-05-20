#!/bin/bash
set -euo pipefail

PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAGE="${1:-nav.html}"
URL="http://localhost:${PORT}/${PAGE}"
SERVER_PID=""

cleanup() {
  if [ -n "${SERVER_PID}" ]; then
    echo ""
    echo "서버 종료중..."
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# 포트 사용 중인지 (bash 내장 /dev/tcp)
if (echo > /dev/tcp/localhost/${PORT}) 2>/dev/null; then
  echo "포트 ${PORT}에 이미 서버가 떠있습니다. 기존 서버 그대로 사용."
else
  echo "백엔드 시작 (포트 ${PORT}, 디렉토리 ${SCRIPT_DIR})"
  cd "${SCRIPT_DIR}"
  python3 -m http.server "${PORT}" > /dev/null 2>&1 &
  SERVER_PID=$!

  # 서버 준비 대기 (최대 ~2초)
  for i in {1..20}; do
    if (echo > /dev/tcp/localhost/${PORT}) 2>/dev/null; then break; fi
    sleep 0.1
  done
fi

# 크롬 새 탭으로 열기 (없으면 chromium → xdg-open 폴백)
if command -v google-chrome > /dev/null 2>&1; then
  google-chrome --new-tab "${URL}" > /dev/null 2>&1 &
elif command -v chromium-browser > /dev/null 2>&1; then
  chromium-browser --new-tab "${URL}" > /dev/null 2>&1 &
elif command -v chromium > /dev/null 2>&1; then
  chromium --new-tab "${URL}" > /dev/null 2>&1 &
elif command -v xdg-open > /dev/null 2>&1; then
  xdg-open "${URL}" > /dev/null 2>&1 &
else
  echo "브라우저 자동 열기 실패. 수동으로 ${URL} 여세요."
fi

echo ""
echo "Running: ${URL}"
echo "Ctrl+C to stop"
echo ""

# 우리가 띄운 서버일 때만 wait
if [ -n "${SERVER_PID}" ]; then
  wait "${SERVER_PID}"
fi
