# Emergency Vehicle Warning System

응급차량의 신속한 출동을 위해 응급차량이 자차 **동일 차선 후방**에 있을 때, 경고를 띄워주는 GPS 기반 프로젝트입니다.

브라우저(Kakao Map) 위에 두 차량의 궤적 / 차선 / 전방 판정 영역을 실시간 재생합니다.

### 시현 영상

<table>
<tr>
<td align="center">
  <video src="https://github.com/user-attachments/assets/2525e61b-6bb4-47dc-ae84-fd577072576e";>
</td>
</tr>
</table>

### 데이터셋
두 개의 데이터셋이 포함되어 있습니다.

| Site | 위치 | 데이터 종류 | 길이 |
|---|---|---|---|
| **ochang** | 충북대학교 오창캠퍼스 C-TRACK 외곽순환도로 | u-blox ZED-F9P, NTRIP(NGII) — 실측 **DGPS** | ~32 s |
| **cbnu** | 충북대학교 본캠퍼스 | u-blox ZED-F9P 이용 도보 측정 RTK-GPS 데이터 | ~4 s |


---

## Quick start

```bash
# 1) Kakao Map JS key 세팅
cp config.example.js config.js
# config.js 를 열어 YOUR_KAKAO_JS_KEY_HERE 를 본인 키로 교체

# 2) 데모 실행 (정적 서버 + 브라우저)
./start.sh                  # nav.html (재생/일시정지/슬라이더)
./start.sh editor.html      # 차선/도로 manual 편집기
```

Kakao 개발자 콘솔의 **플랫폼 → Web → 사이트 도메인** 에 `http://localhost:8000` 을 등록해두지 않으면 SDK 가 실패합니다.

### 파이썬 환경

```bash
pip install numpy pandas scipy pyproj pillow matplotlib
```

`python3 -m http.server 8000` 만 띄우면 데모는 동작하므로 파이썬 의존성은 **파이프라인을 재실행할 때만** 필요합니다.

---

## Pipeline overview

```
bag_extract CSV ──► pipeline.py ──► web/sites/<site>/{frames,lanes,raw_trips,final_*}.json/csv
                       │                          │
                       │                          ▼
                       │                  nav.html (Kakao Map)
                       │
                       └─► build_ego_camera.py ──► web/sites/<site>/ego_camera/*.jpg
                                                   web/sites/<site>/ego_camera_index.json
```

`pipeline.py` 안의 `SITES` dict 가 각 사이트의 입력 경로, lane 폭, 데모용 속도 스케일, KF 사용 여부를 한 곳에서 관리합니다.
새 사이트를 추가하려면 이 dict 에 한 항목만 더하면 됩니다.

핵심 처리 단계:

1. **UTM 변환** (EPSG:32652) — 차선폭(미터) 단위 판정.
2. **U-turn 탐지 및 trim** (왕복 데이터에서 outbound 만).
3. **Bias 보정** — `data/sites/<site>/final.json` 의 `(dx_east, dy_north)` 가 위성 영상 위에서 manual 으로 정해진 2D shift.
4. **CV-Kalman + RTS smoother** (`gps_kf.py`) — heading 표준편차 1.58° → 0.41° (오창 기준 -74%).
5. **Lane geometry** — bag1 의 forward 궤적을 R 차선, 복귀 궤적을 L 차선으로 재샘플링하거나, `editor.html` 로 직접 그린 centerline 사용.
6. **Per-frame 판정** — ego 의 10 m 전방 직사각형 안에 emergency 가 들어왔는지로 `same_lane_ahead` boolean 산출.

자세한 KF 설계 근거(왜 CV, 왜 EKF/CTRV 가 아닌가, NIS 검증)는 `gps_kf.py` 의 docstring 과 `FilterConfig` 주석에 적어두었습니다.

---

## Reproducing the pipeline

```bash
# 사이트 단위로 처리 (lanes.json + frames.json 갱신) — 프로젝트 root 에서 실행
python3 src/pipeline.py --site ochang
python3 src/pipeline.py --site cbnu

# ego 카메라 패널 갱신 (원본 dashcam 이 있을 때만 — git 제외분)
python3 src/build_ego_camera.py --site ochang
```

`final.json` 의 bias 와 trim 값은 위성 영상을 보면서 `editor.html` 에서 시각적으로 맞춘 값입니다.
다른 데이터셋으로 옮길 때는 이 두 값만 다시 잡아주면 같은 파이프라인이 그대로 돕니다.

---

## Notes on the GPS measurement

오창 측정에서 RTK Fixed 가 안 잡힌 이유와 그 영향은 다음과 같이 정리됩니다.

- NTRIP RTCM3 (NGII) 수신은 정상 (`diffSoln=1` 전 구간), 그러나 carrier-phase ambiguity 미해결 (`carrSoln=0`).
- `hAcc` median ≈ 1 m → DGPS-급. 차선폭(3.5 m) 단위 판정의 한계점에 있음.
- 1 m 오차 분해: 고주파 잡음 ≈ 0.6 cm, slow bias ≈ 나머지 전부.
- → KF 는 잡음에만 효과 있음. **slow bias 는 위성 영상 기반 manual 2D shift** 로 별도 처리.
- 다음 측정 시 권고: u-center 에서 `RXM-RAWX` / `RXM-SFRBX` 활성화 (PPK fallback 확보).

코드는 RTK Fixed 입력을 그대로 받도록 작성돼 있어, 더 좋은 측정값이 확보되면 데이터만 갈아끼우면 됩니다.

---

## Acknowledgments

충북대학교 위성항법시스템 수업 프로젝트로 시작했으며, u-blox ZED-F9P + NTRIP setup을 활용했습니다.
