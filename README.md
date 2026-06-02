# Ollama PDF RAG Chatbot

로컬 PDF를 기반으로 한국어 질의응답을 수행하는 Ollama + Chroma 기반 RAG 챗봇입니다.

PDF 내용은 외부 API로 보내지 않고 로컬 LLM(Ollama)과 로컬 임베딩(KURE-v1)으로 처리하므로, 사내 문서나 개인 자료처럼 외부 전송이 어려운 PDF를 로컬에서 다룰 수 있습니다. Chroma vector search와 `kiwipiepy` 기반 BM25를 결합한 hybrid retrieval, 그리고 metadata 기반 출처 자동 부착으로 환각(hallucination)을 줄이고 답변 근거를 분명하게 표시합니다. CLI와 Chainlit 기반 브라우저 UI를 모두 지원합니다.

> ⚠️ **로컬 전용 프로젝트입니다.**
> 본인 PC에서 단일 사용자가 직접 실행하는 용도로 설계되었으며, **인증·접근 제어 기능이 없습니다.** Chainlit UI는 기본적으로 `localhost(127.0.0.1)`에만 바인딩되므로 로컬 실행 시에는 본인만 접속할 수 있습니다.
>
> - 공용/외부 네트워크에 노출하지 마세요. `chainlit run app.py --host 0.0.0.0` 처럼 외부 host로 바인딩하거나 터널·리버스 프록시로 공개하면, **인증 없이 누구나 문서 질의·업로드가 가능**해집니다.
> - 여러 사용자에게 서비스하려면 별도의 인증, origin 제한, 업로드 크기 제한, 에러 메시지 처리 등을 직접 추가해야 합니다.
> - PDF·인덱스·벡터 DB(`data/pdfs/`, `indexes/`, `chroma_db/`)는 로컬에만 저장되며 외부로 전송되지 않습니다.

## 목차

- [시스템 요구사항](#시스템-요구사항)
- [빠른 시작](#빠른-시작)
- [실행 방법](#실행-방법)
- [Chainlit UI](#chainlit-ui)
- [데이터 흐름](#데이터-흐름)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [평가 및 성능 측정](#평가-및-성능-측정)
- [주요 설정](#주요-설정)
- [트러블슈팅](#트러블슈팅)
- [Known Limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## 시스템 요구사항

| 항목 | 권장 사양 |
|---|---|
| Python | 3.10 이상 |
| 메모리(RAM) | 8GB 이상, 16GB 권장 (Ollama 7.8b 모델 + KURE-v1 + bge-reranker를 함께 사용할 경우 메모리 사용량이 커질 수 있음) |
| 디스크 | 모델 캐시 약 5GB 이상 여유 (KURE-v1 ≈ 2.3GB, bge-reranker-v2-m3 ≈ 2.3GB, Ollama 모델 별도) |
| Ollama | Ollama 앱 또는 로컬 서비스가 실행 중이어야 함 (`ollama serve`) |
| GPU (선택) | CUDA 사용 시 `EMBED_DEVICE=cuda`, `RERANKER_DEVICE=cuda`로 가속 |

GPU 가속 예시:

```bash
EMBED_DEVICE=cuda RERANKER_DEVICE=cuda python -m src.main index
```

기본값은 `cpu`이며, Mac/Linux/Windows 모두 CPU만으로 동작합니다.

## 빠른 시작

아래 순서대로 실행하면 PDF 인덱싱 후 첫 질문까지 바로 확인할 수 있습니다.

### 1. 프로젝트 받기

```bash
git clone https://github.com/jinyoo1021/ollama_RAG.git
cd Ollama_RAG
```

### 2. 가상환경 및 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell에서는 다음 명령을 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ollama 설치 및 모델 준비

[https://ollama.com](https://ollama.com)에서 Ollama를 설치한 뒤, Ollama 앱을 실행하거나 터미널에서 로컬 서비스를 실행합니다.

```bash
ollama serve
```

Ollama 앱이 이미 실행 중이면 `ollama serve`는 생략할 수 있습니다.

별도 터미널에서 사용할 모델을 받습니다.

```bash
ollama pull exaone3.5:7.8b
```

저사양 환경에서는 아래 모델로 먼저 테스트할 수 있습니다.

```bash
ollama pull exaone3.5:2.4b
```

### 4. PDF 넣기

질문할 PDF 파일을 아래 폴더에 넣습니다.

```text
data/pdfs/
```

GitHub에는 PDF 파일이 포함되지 않으므로, 실행 전에 직접 PDF를 추가해야 합니다.

### 5. 인덱싱

```bash
python -m src.main index
```

> 처음 실행할 때는 임베딩 모델(KURE-v1, 약 2.3GB)을 HuggingFace에서 자동으로 다운로드합니다. 네트워크 상태에 따라 첫 인덱싱은 수 분에서 십수 분 걸릴 수 있고, 이후 실행에서는 로컬 캐시(`~/.cache/huggingface/`)를 사용하므로 빠르게 시작합니다. Reranker 모델(bge-reranker-v2-m3, 약 2.3GB)은 `USE_RERANKER=true`로 실행할 때 다운로드됩니다.

### 6. 첫 질문

```bash
python -m src.main --reuse-index ask "문서의 핵심 내용은 무엇인가요?"
```

## 실행 방법

빠른 시작에서 한 번 인덱싱했다면 아래 명령으로 재사용할 수 있습니다. 대부분의 사용 상황에서는 문서가 바뀌지 않았다면 `--reuse-index`로 인덱싱 시간을 줄이는 것이 기본입니다.

> **주의**: CLI 기본값은 매번 재인덱싱(`config.py`의 `REBUILD_INDEX=True`)이므로 `--reuse-index`를 빼면 실행할 때마다 인덱스를 새로 만듭니다. 문서가 바뀌지 않았다면 아래 명령에서처럼 `--reuse-index`를 붙이세요.

### PDF가 추가/삭제/교체된 경우 재인덱싱

```bash
python -m src.main index
```

### 단일 질문

```bash
python -m src.main --reuse-index ask "[질문]"
```

출력 예시:

```text
$ python -m src.main --reuse-index ask "근로기준법의 적용범위는?"
근로기준법은 상시 5명 이상의 근로자를 사용하는 모든 사업 또는 사업장에 적용됩니다.
다만 동거하는 친족만을 사용하는 사업과 가사 사용인에게는 적용되지 않습니다.

출처: 근로기준법 p.1, p.2
```

답변 본문에는 페이지 번호가 들어가지 않고, 마지막 줄에 `출처: 문서명 p.N` 형식으로 metadata 기반 출처가 자동 부착됩니다.

### 대화형 chat

```bash
python -m src.main --reuse-index chat
```

멀티턴 chat은 최근 대화 맥락을 사용해 짧은 후속 질문의 검색 query를 보강합니다. 자세한 구현 배경은 [Phase 3 기록](./DEVELOPMENT_LOG.md#phase-3-대화형-기능-구현)을 참고하세요.

### Retrieval 디버그

검색 후보와 점수를 확인하려면 debug retrieval을 켭니다.

```bash
python -m src.main --reuse-index ask --debug-retrieval "근로기준법 출석의 의무"
```

### Reranker 켜기

```bash
USE_RERANKER=true python -m src.main --reuse-index ask "질문"
```

## Chainlit UI

브라우저 기반 UI를 실행합니다.

```bash
chainlit run app.py
```

접속 주소:

```text
http://localhost:8000
```

UI 실행 시 인덱스를 강제로 다시 만들고 싶다면:

```bash
CHAINLIT_REBUILD_INDEX=true chainlit run app.py
```

UI에서 PDF를 추가하려면 채팅창에 아래 명령을 입력합니다.

```text
/upload
```

PDF를 메시지에 첨부해도 업로드 처리됩니다. 업로드한 PDF는 `data/pdfs/`에 저장되고, 전체 PDF 기준으로 인덱스를 다시 생성합니다. 문서 묶음이 바뀌면 이전 대화 history는 초기화됩니다.

답변의 출처는 본문 마지막에 `문서명 p.N` 형식으로 표시됩니다. UI 구현과 업로드 흐름은 [Phase 6 완료 정리](./DEVELOPMENT_LOG.md#phase-6-완료-정리)를 참고하세요.

## 데이터 흐름

```text
PDF → Markdown 추출 → 청킹 → 임베딩 → Chroma 인덱싱
                                      ↓
              Ollama 답변 ← Hybrid retrieval(Vector + BM25, optional Reranker)
                                      ↓
                              metadata 기반 출처 부착
```

각 단계의 구현 배경과 결정 과정은 [개발 일지 요약](./DEVELOPMENT_LOG.md#요약)에 정리되어 있습니다. PDF 로딩·청킹·기본 RAG는 [Phase 2 기록](./DEVELOPMENT_LOG.md#phase-2-기본-rag-파이프라인-구현), hybrid·reranker는 [Phase 4 기록](./DEVELOPMENT_LOG.md#phase-4-정확도-개선)을 참고하세요.

## 주요 기능

### 인덱싱
- PDF를 `pymupdf4llm`으로 페이지 단위 Markdown으로 추출하고 페이지 metadata 보존
- 헤더 기반 분할 + 문자 단위 RecursiveCharacterTextSplitter로 청킹
- Chroma 영속 벡터 저장소와 BM25용 chunk 직렬화(`indexes/chunks.json`) 동시 생성

### 검색
- Chroma 기반 vector search
- `kiwipiepy` 형태소 분석 + `rank-bm25` 기반 한국어 BM25 검색
- BM25 + vector 가중 결합 hybrid retrieval (`BM25_WEIGHT`/`VECTOR_WEIGHT`로 조정)
- `USE_RERANKER=true`로 cross-encoder reranker(bge-reranker-v2-m3) optional 실행
- 법률 문서 cross-law 참조 추적 (다른 법령 인용 조항 자동 보강)

### 답변과 UI
- 단일 질문 CLI (`ask`)와 멀티턴 대화형 chat (`chat`)
- Ollama 스트리밍 답변 + 모델이 생성한 inline 출처 표기 실시간 제거
- metadata 기반 `출처: 문서명 p.N` 자동 부착 (LLM 환각 출처 차단)
- Chainlit 브라우저 UI와 `/upload` 명령으로 PDF 추가 및 전체 재인덱싱

### 평가와 운영
- `--debug-retrieval`로 vector/BM25/reranker 후보와 점수 출력
- LLM 없이 retrieval/source 정확도를 측정하는 평가 script
- 단계별 latency 측정 script (retrieval / prompt / source / optional LLM)
- reranker on/off 비교, chunk 크기 튜닝 script

Hybrid retrieval과 reranker 관련 상세 내용은 [Phase 4 기록](./DEVELOPMENT_LOG.md#phase-4-정확도-개선)을 참고하세요. 주요 문제 해결 히스토리는 [트러블 슈팅](./DEVELOPMENT_LOG.md#트러블-슈팅-내용)에 정리되어 있습니다.

## 프로젝트 구조

```text
.
├── README.md
├── DEVELOPMENT_LOG.md
├── docs/
│   └── initial-plan.md
├── requirements.txt
├── config.py
├── app.py
├── src/
│   ├── main.py
│   ├── rag_service.py
│   ├── chat_manager.py
│   ├── llm_client.py
│   ├── pdf_loader.py
│   ├── indexer.py
│   ├── evaluation.py
│   └── retrieval/
│       ├── hybrid.py
│       ├── bm25.py
│       ├── reranker.py
│       ├── legal_references.py
│       ├── source_resolver.py
│       ├── document_utils.py
│       └── debug.py
├── data/
│   └── pdfs/
├── chroma_db/
├── indexes/
├── scripts/
│   ├── evaluate_retrieval.py
│   ├── measure_latency.py
│   ├── compare_reranker.py
│   └── tune_chunks.py
└── tests/
    ├── test_cases.py
    └── eval_cases.py
```

`src/retrieval/`은 검색, reranker, 법률 참조 추적, 출처 선택, debug 출력을 담당합니다. 패키지 분리 배경은 [Phase 7 리팩터링 기록](./DEVELOPMENT_LOG.md#17-phase-7-retrieval-패키지-분리-리팩터링)을 참고하세요.

## 평가 및 성능 측정

Retrieval 정확도와 출처 선택 품질을 LLM 호출 없이 확인합니다.

```bash
python scripts/evaluate_retrieval.py --case-set legal --repeat 1
python scripts/evaluate_retrieval.py --case-set all --repeat 2
```

Latency를 retrieval, prompt build, source formatting, optional LLM generation으로 나누어 측정합니다.

```bash
python scripts/measure_latency.py --case-set legal --limit 2 --repeat 1 --warmup 0
python scripts/measure_latency.py --query "근로기준법 출석의 의무" --include-llm
```

Reranker on/off 품질과 속도를 비교합니다.

```bash
python scripts/compare_reranker.py --repeat 1
```

청크 크기 후보를 비교합니다.

```bash
python scripts/tune_chunks.py --chunk-sizes 800 1200 1500 2000
```

평가와 latency script 정리 내용은 [Phase 7 평가 기록](./DEVELOPMENT_LOG.md#16-phase-7-평가latency-스크립트-정리)을 참고하세요.

## 주요 설정

주요 설정은 [config.py](./config.py)에 있습니다.

| 설정 | 기본값 | 의미 |
|---|---|---|
| `OLLAMA_MODEL` | `exaone3.5:7.8b` | 답변 생성에 사용할 Ollama 모델 |
| `EMBEDDING_MODEL` | `nlpai-lab/KURE-v1` | PDF chunk 임베딩 모델 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | optional reranker 모델 |
| `CHUNK_SIZE` | `1200` | 기본 청크 크기(문자 단위) |
| `CHUNK_OVERLAP` | `120` | 청크 overlap(문자 단위) |
| `RETRIEVAL_TOP_K` | `10` | vector/BM25 검색 후보 수 |
| `RERANK_TOP_K` | `3` | reranker 사용 시 상위 후보 수 |
| `BM25_WEIGHT` | `0.4` | hybrid 점수에서 BM25 비중 |
| `VECTOR_WEIGHT` | `0.6` | hybrid 점수에서 vector 비중 |
| `SOURCE_SCORE_MARGIN` | `0.05` | 출처 후보로 포함할 점수 마진 |
| `MAX_HISTORY_TURNS` | `5` | chat에서 유지할 최근 대화 턴 수 |
| `REBUILD_INDEX` | `True` | CLI 기본 동작: True이면 `--reuse-index`를 안 붙였을 때 매번 재인덱싱 |

### 환경 변수

실행 시 셸에서 지정하는 런타임 옵션입니다. `config.py` 상수와 달리 코드를 수정하지 않고 켜고 끌 수 있습니다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `USE_RERANKER` | `false` | `true`로 설정하면 cross-encoder reranker를 켭니다. 처음 `true`로 실행할 때 bge-reranker-v2-m3(약 2.3GB)를 다운로드합니다. |
| `EMBED_DEVICE` | `cpu` | 임베딩 모델 디바이스. 현재 환경에서 지원되는 `cuda`/`mps`/`cpu` 중 하나를 직접 지정합니다. |
| `RERANKER_DEVICE` | `EMBED_DEVICE`를 따름 | reranker 디바이스를 별도로 지정할 때 사용합니다. |
| `CHAINLIT_REBUILD_INDEX` | `false` | `true`로 설정하면 Chainlit UI 시작 시 인덱스를 강제로 재생성합니다. |

사용 예:

```bash
USE_RERANKER=true EMBED_DEVICE=cuda python -m src.main --reuse-index ask "질문"
CHAINLIT_REBUILD_INDEX=true chainlit run app.py
```

## 트러블슈팅

자주 발생하는 문제와 해결 방법입니다. 보다 상세한 트러블슈팅 히스토리는 [DEVELOPMENT_LOG.md의 트러블 슈팅 섹션](./DEVELOPMENT_LOG.md#트러블-슈팅-내용)을 참고하세요.

### `Error: No PDF files found in data/pdfs`

`data/pdfs/` 디렉토리에 PDF가 한 개도 없을 때 발생합니다. 폴더를 만들고 PDF를 넣은 뒤 다시 실행하세요.

```bash
mkdir -p data/pdfs
# data/pdfs/ 에 PDF 추가 후
python -m src.main index
```

### `httpx.ConnectError` 또는 `Connection refused`

Ollama 로컬 서비스가 실행되고 있지 않을 때 발생합니다. 별도 터미널에서 서비스를 띄우거나 Ollama 앱을 실행하세요.

```bash
ollama serve
```

### `model 'exaone3.5:7.8b' not found`

해당 모델이 로컬에 받아져 있지 않습니다. `ollama pull`로 모델을 받으세요.

```bash
ollama pull exaone3.5:7.8b
```

다른 모델을 쓰고 싶다면 [config.py](./config.py)의 `OLLAMA_MODEL` 값을 변경하면 됩니다.

### 첫 인덱싱이 매우 오래 걸림

처음 실행 시 임베딩 모델(KURE-v1, 약 2.3GB)을 HuggingFace에서 다운로드합니다. 네트워크 상태에 따라 수 분에서 십수 분이 걸릴 수 있습니다. 이후 실행에서는 로컬 캐시(`~/.cache/huggingface/`)를 사용하므로 빨라집니다.

### 메모리 부족(OOM)

Ollama 7.8b 모델, KURE-v1, bge-reranker를 함께 사용하면 메모리 사용량이 커집니다. 메모리가 부족하면 `USE_RERANKER`를 끄거나(`false`) 더 작은 Ollama 모델(`exaone3.5:2.4b`)을 사용해 보세요.

## Known Limitations

현재 알려진 제약 사항과 향후 개선 방향입니다.

- **인덱스 증분 업데이트 미지원**: PDF가 한 개라도 바뀌면 전체 인덱스를 다시 만들어야 합니다. 대용량 corpus에서는 시간이 오래 걸릴 수 있습니다.
- **파일 hash 기반 중복 인덱싱 방지 없음**: 같은 PDF를 이름만 바꿔 업로드해도 별도 chunk로 색인됩니다. 같은 문서가 출처에 중복으로 나타날 수 있습니다.
- **Chroma collection 관리 명령 부재**: collection 삭제/이름 변경/통계 확인 CLI가 없어 직접 `chroma_db/` 디렉토리를 다루어야 합니다.
- **Preliminary source 노이즈**: 답변 생성 전 단계에서 모델에 전달되는 출처 후보가 실제 근거와 어긋날 때가 있어, 본문은 정확해도 후보 메시지가 혼란을 줄 수 있습니다.
- **복합 질문 answer-aware source 평가 보강 필요**: 한 질문에 여러 근거가 필요한 경우, 출처 선택 로직이 한쪽 근거만 선호할 수 있습니다.
- **LLM 포함 latency 리포트 보강 필요**: `--include-llm`로 생성 시간을 측정할 수 있지만, 대표 케이스 기반 비교 리포트는 더 보강할 필요가 있습니다.

## Acknowledgements

이 프로젝트의 설계, 구현, 문서 정리 과정에서 Codex와 Claude를 보조 도구로 활용했습니다.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
