# DRIFT-Demo

[English](./README.md) · **한국어**

**[DRIFT](https://github.com/TaewoooPark/DRIFT) P2P 추론 한 번을 두 화면으로 보여주는 비주얼 데모 — "For Tokens" 경제의 양면을 동시에.**

하나의 모델이 두 DRIFT 워커에 레이어 단위로 쪼개져(peer-to-peer chain + weightless head) 돌아가고, 거래의 양쪽이 각자 풀스크린 화면을 가집니다. 모든 패널은 추상화가 아니라 네트워크의 **실제 내부**를 보여줍니다:

| 화면 | 누구 | 무엇이 보이나 |
|---|---|---|
| **A · consumer** (`/a`) | 질문하는 쪽 | 채팅창; **레이어별 ‖Δh‖** — 자기 몫의 디코더 레이어(`[0:14)`)가 이 토큰의 표현을 실제로 얼마나 고쳐 썼는지; 와이어를 건너 떠나는 **residual stream 그 자체**(토큰당 1536개 fp16 값을 스크롤 히트맵으로); 네트워크가 저울질한 **다음 토큰 후보** 티커; 실제 스텝이 한 줄씩 흐르는 터미널식 **연산 로그** |
| **B · provider** (`/b`) | 기여하는 쪽 | 자기 절반(`[14:28)`)의 동일한 라이브 내부 — **도착하는** residual stream(A 화면을 떠난 것과 비트 동일), 레이어 기록 바, **이 노드가 직접 계산하는 top-k lm_head 확률**(헤드는 무게 0), 로그에 찍히는 홉마다의 **Ed25519 서명 영수증**, 그리고 올라가는 기여 정산(**layer·tokens**) |

노트북 두 대를 나란히 두면(A 왼쪽, B 오른쪽) consumer의 패킷은 오른쪽 모서리로 빠져나가고 provider의 패킷은 왼쪽에서 들어옵니다 — 그리고 두 히트맵이 **같은 세로줄**을 그립니다. 같은 바이트이기 때문입니다: 두 머신이 하나의 forward pass를 굴리고 있다는 시각적 증명.

**모든 픽셀은 실물입니다.** 이 데모는 DRIFT 소스를 한 줄도 수정하지 않고, 어떤 데이터도 시뮬레이션하지 않습니다: 순정 DRIFT 워커를 프로세스 시작 시 몽키패칭(`TorchShardEngine.load/forward/head_argmax`, `Node.handle`, `Node._relay`)과 디코더 레이어별 read-only PyTorch forward hook으로만 계측하고, 전부 아웃오브밴드 fire-and-forget UDP 이벤트로 내보냅니다. 수학은 그대로입니다 — 다음 토큰은 순정 코드와 같은 수식으로 쓴 단 한 번의 `lm_head` 계산에서 나오고, 데모의 greedy 출력이 순정 경로와 토큰 단위로 동일함을 검증했습니다(표시용 추출 비용은 머신에 따라 토큰당 대략 10–20 ms, `DRIFT_DEMO_TOPK=0`으로 top-k 탭을 끌 수 있음). 화면의 영수증 해시는 헤드가 실제로 검증하는 그 영수증이며, 실행은 저널로 남아 `drift ledger`로 데모 자체를 감사할 수 있습니다. 라이브 검증: A의 나가는 히트맵 열과 B의 들어오는 열이 **30/30 스텝 완전 일치** — DRIFT의 패리티 게이트가 증명하는 그대로, fp16 와이어 왕복은 무손실입니다.

## 화면 읽는 법

아래 두 그림은 라이브 로컬 실행 중에 캡처한 것이며, 번호는 이미지 속 배지와 일치합니다.

### A · consumer — `/a`

![view A, annotated](docs/view-a.png)

1. **트랜스크립트** — 대화 내용. 실제 `drift run` REPL과 같은 문법(`you ›` / `drift ›`)이며, 체인 왕복이 한 번 끝날 때마다 토큰이 하나씩 도착합니다.
2. **후보 티커** — 최신 스텝에서 네트워크가 저울질한 다음 토큰 후보들과 실제 확률(tail 노드 자신의 `lm_head` softmax).
3. **프롬프트** — 여기 입력하면 `POST /api/generate`로 실제 오케스트레이터가 구동됩니다.
4. **레이어별 ‖Δh‖** — 이 머신이 든 디코더 레이어 `[0:14)`. 바 높이 = 그 레이어가 방금 이 토큰의 히든 표현을 얼마나 고쳐 썼는지 (read-only forward hook 실측; 세그먼트 미터가 토큰마다 스냅).
5. **떠나는 residual stream** — 와이어를 실제로 건너는 히든 스테이트: 마지막 위치의 1536개 fp16 값을 평균-|활성값| 128버킷으로 줄여 토큰당 세로줄 하나로, 1-bit 렌더링(Bayer 오더드 디더링 — 흰 픽셀의 밀도가 곧 크기). 노드 B는 정확히 이 바이트를 받습니다.
6. **와이어** — 채워진 패킷은 노드 B로 떠나는 ~3.0 KB의 히든 스테이트, 테두리만 있는 패킷은 집으로 돌아오는 토큰 id 하나. 이 비대칭이 곧 weightless-head 설계입니다: 텐서는 노드 사이를 흐르고, 헤드에는 정수만 닿습니다.
7. **연산 로그** — 이 머신의 실제 스텝이 한 줄씩: `<<` 수신(와이어에서 내려온 바이트), `::` 연산(레이어 범위 + ms), `>>` 송신, `OK` 헤드가 Ed25519 영수증 체인 전체를 라이브로 검증한 기록(해시 프리픽스 표시).
8. **세션 통계** — tok/s, ms/token, 토큰당 와이어 바이트, 지금까지 검증된 영수증 수, 모델.

### B · provider — `/b`

![view B, annotated](docs/view-b.png)

1. **도착하는 residual stream** — A의 나가는 패널과 같은 세로줄. 같은 바이트이기 때문입니다(테스트에서 30/30 스텝 동일 확인): 화면은 둘, forward pass는 하나.
2. **레이어별 ‖Δh‖** — 이 머신의 절반, 레이어 `[14:28)`. 의미는 A의 미터와 동일.
3. **와이어** — 노드 A에서 히든 스테이트가 들어오고, 토큰 id 하나가 헤드 쪽으로 나갑니다.
4. **연산 로그** — 이 머신의 스텝들. `#` 서명 줄이 포함됩니다: 이 노드가 자기가 계산한 모든 홉에 대해 서명하는 Ed25519 영수증(`in`/`out` 해시 + 서명 프리픽스 — 렛저가 정산하는 바로 그 영수증).
5. **기여(Contribution)** — 렛저 정산: **layer·tokens**(든 레이어 수 × 나른 토큰 수, M13의 정산 단위), 나른 토큰 수, 서빙한 세션 수, 이 노드의 신원 키.
6. **검증 상태** — 헤드의 토큰별 검사가 통과하는 동안 `ALL HOPS VERIFIED` 유지; 서명/해시 인접성/앵커 검사 하나라도 깨지는 순간 반전 깜빡임 `SUSPECT …`로 뒤집힙니다.
7. **다음 토큰 후보** — **이 노드가 돌리는** `lm_head`에서 나온 실제 확률(헤드는 가중치 0): `>`가 실제 선택된 토큰, `█░` 미터가 softmax.
8. **세션 통계** — 이번 실행 토큰 수, ms/token, 토큰당 와이어, 영수증, 모델.

## 실행

Python **3.12**와 [`uv`](https://github.com/astral-sh/uv)가 필요합니다.

```bash
bash scripts/setup.sh          # DRIFT를 vendor/에 클론, .venv 구성
.venv/bin/python -m demo       # 로컬 워커 2개 스폰, /a /b 자동 오픈
```

그다음 A 화면에 프롬프트를 입력하세요. 첫 기동은 모델 샤드를 로드하는 데 ~10–60초 걸리고, 캐시가 없는 머신이라면 먼저 Hugging Face 캐시로 모델을 내려받습니다(기본 Qwen 기준 ~3 GB). 네트워크가 조립되면 오버레이가 걷힙니다.

```
http://127.0.0.1:8800/a   consumer
http://127.0.0.1:8800/b   provider
```

옵션:

```
python -m demo --nodes 3            # 워커 추가 (N번째 화면: /b?node=2)
python -m demo --model <hf-id>      # DRIFT가 도는 모델 아무거나 (기본 Qwen2.5-1.5B-Instruct)
python -m demo --max-new-tokens 400
python -m demo --no-browser --port 8800
```

## 모델

데모에는 모델에 대한 하드코딩이 **하나도 없습니다**: ‖Δh‖ 훅은 `engine.layers`에 든 것을 그대로 순회하고, 히트맵은 어떤 hidden 크기든 128버킷으로 adaptive pooling하며, top-k 탭은 인트로스펙션된 `lm_head`와 모델 자신의 토크나이저를 씁니다. 따라서 DRIFT가 돌릴 수 있는 모델이면 그대로 여기서도 돕니다:

```bash
python -m demo --model Qwen/Qwen2.5-7B-Instruct
python -m demo --model google/gemma-4-E2B-it
```

제약은 데모가 아니라 DRIFT의 것입니다: 설치된 `transformers`가 지원하는 decoder-only Hugging Face causal LM, 그리고 워커들의 합산 메모리에 들어가는 fp16 가중치. 레이어 패널·와이어 크기·분할 지점은 전부 로드된 모델에서 스스로 다시 유도됩니다.

## 실행 감사하기

헤드는 검증된 모든 영수증을 `.state/journal-<ts>.jsonl`에 저널링합니다:

```bash
.venv/bin/drift ledger .state/journal-*.jsonl --verify
```

로컬 워커 두 개는 **서로 다른** Ed25519 신원(`.state/node{0,1}.identity`)으로 서명하므로, 한 머신에서도 정산표에 기여자 두 명이 나옵니다.

## 구조

```
demo/node_main.py     순정 `drift node` + 계측 (mDNS/gossip 없음 — 로컬 데모)
demo/instrument.py    몽키패치 모음: 레이어별 ‖Δh‖ 훅, residual stream 다운샘플,
                      tail 자신의 lm_head에서 top-k, 스텝 도착, 연산 타이밍, p2p 릴레이
demo/head.py          weightless(thin) 헤드 + 스텝 단위 디코드 루프, 토큰마다 이벤트
demo/events.py        fire-and-forget UDP JSON 방출기 (아웃오브밴드, 논블로킹)
demo/server.py        stdlib HTTP: /a /b, SSE /events, POST /api/generate, /api/state
demo/__main__.py      런처: 워커 스폰 → chain+thin 헤드 조립 → 서빙
demo/static/          두 화면 (순수 HTML/CSS/JS, 빌드 스텝 없음)
tests/                torch 불필요 스모크 테스트: 상태 fold, 정확히-한-번 attach,
                      API 검증 (`python -m unittest discover -s tests`)
```

생성 토큰 하나당 토폴로지 (chain + thin head, M7/M10):

```
head ──ids──▶ n0 [0:14) ──hidden 3.1 KB──▶ n1 [14:28) ──token──▶ head
              └─ 영수증 서명                └─ 영수증 서명         └─ 둘 다 라이브 검증
```

**페일오버도 데모 루프로 검증했습니다.** 생성 중에 워커를 죽여도 세션이 살아남습니다(M9): 헤드가 생존자 위로 재분할·재생하고 플랜을 다시 방송해 화면 패널이 리빌드됩니다 — 토큰 ~30 시점 `SIGKILL` 실측에서 완성된 텍스트가 무중단 실행과 **비트 동일**했고, 렛저에 그 서사가 그대로 남습니다(생존자의 range가 `[0:14),[0:28)`로 넓어지고 지분이 상승).

DRIFT 자체는 `vendor/DRIFT`에 읽기 전용으로 벤더링됩니다 — gitignore 대상이며, `scripts/setup.sh`가 DRIFT **`v1.0.0`** 태그로 고정합니다(데모가 drift 내부에 훅을 걸므로 업그레이드는 의도적으로: `rm -rf vendor/DRIFT && DRIFT_REF=<tag> bash scripts/setup.sh`).
