# Emergency Vehicle Same-Lane Warning

긴급차량이 자차 **같은 차선 뒤**에 있을 때만 경고를 띄우는 GPS-기반 데모입니다.
브라우저(Kakao Map) 위에 두 차량의 궤적 / 차선 / 전방 판정 영역을 실시간 재생합니다.

두 개의 데이터셋이 포함되어 있습니다.

| Site | 위치 | 데이터 종류 | 길이 |
|---|---|---|---|
| **ochang** | 충북대학교 오창캠퍼스 C-TRACK 외곽순환도로 | u-blox ZED-F9P 듀얼 rover, NTRIP(NGII) — 실측 **DGPS** | ~32 s |
| **cbnu** | 충북대학교 본캠퍼스 | 직접 측정 후 RTK Fixed 가정으로 재포맷한 차선변경 시나리오 | ~4 s |

오창은 RTK가 안 잡혔던 실측 데이터(`carrSoln=0`, hAcc≈1 m)이고,
1 m 오차가 거의 전부 **slow bias**라는 분석 결과를 토대로 위성 영상 기준 수동 2D 정렬을 추가했습니다.
한쪽 한쪽 단순화하지 않고 측정 한계와 그 보완을 명시적으로 분리한 점이 이 프로젝트의 핵심입니다.

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

## Repository layout

```
.
├── nav.html                 # 데모 (재생/슬라이더/차량 마커/판정 박스)
├── editor.html              # centerline / lane / bias 를 위성 영상 위에서 편집
├── config.example.js        # Kakao JS key 템플릿 (config.js 는 .gitignore)
├── start.sh                 # http.server + 브라우저 자동 오픈
│
├── src/
│   ├── pipeline.py          # bag CSV → JSON 단일 진입점 (--site ochang|cbnu)
│   ├── gps_kf.py            # CV-KF + RTS smoother (standalone 실행 가능)
│   └── build_ego_camera.py  # source/camera/ → web/ego_camera/ trim + crop
│
├── data/
│   └── sites/
│       ├── ochang/
│       │   ├── final.json           # manual road geometry + bias + trim
│       │   └── source/
│       │       ├── gps1_extracted/  # emergency vehicle u-blox CSV
│       │       ├── gps2_extracted/  # ego vehicle u-blox CSV
│       │       └── camera/          # git 제외 (원본 dashcam ~700 MB)
│       └── cbnu/
│           ├── final.json
│           └── source/lane{1,2}_extracted/
│
└── web/
    ├── sites.json
    └── sites/
        ├── ochang/{frames,lanes,raw_trips,final_*,ego_camera_index}.json/csv
        │   └── ego_camera/*.jpg     # 데모 영상 패널 (trim + crop 결과)
        └── cbnu/{frames,lanes,...}.json/csv
```

### 데이터 정책

- **포함**: u-blox extract CSV, `final.json`(manual 라벨), pipeline 출력 JSON, ego_camera trim 출력.
- **제외 (`.gitignore`)**: source dashcam JPG 원본(`data/sites/*/source/camera/`), 발표/수업 노트(`data/*.md`), 생성형 figure(`data/figures/`).

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

충북대학교 위성항법시스템 수업 프로젝트로 시작했으며, 자율주행팀 (`Clothoid-R`) 의 u-blox ZED-F9P + NTRIP setup 을 활용했습니다.
