# 개발 일지

작성일: 2026-05-20
업데이트: 2026-05-29

## 요약

Ollama 기반 로컬 PDF RAG 챗봇의 Phase 1부터 Phase 7까지 진행했다.

현재 프로젝트는 `data/pdfs/`에 있는 PDF를 읽고, `pymupdf4llm`로 Markdown을 추출한 뒤, `KURE-v1` 임베딩과 ChromaDB를 사용해 로컬 벡터 인덱스를 생성한다. 검색은 Chroma vector search와 `kiwipiepy` 기반 BM25를 결합한 hybrid retrieval을 사용한다. CLI와 Chainlit UI에서 단일 질문, 대화형 질문, PDF 업로드를 사용할 수 있으며, Ollama의 `exaone3.5:7.8b` 모델을 통해 한국어 답변을 스트리밍으로 출력한다. Phase 7에서는 retrieval/source 평가와 latency 측정 기반을 만들고, retrieval 계열 코드를 `src/retrieval/` 패키지로 분리했다.

중간 검증 과정에서 페이지 출처 오류, 청킹 단위 문제, 멀티턴 검색 맥락 문제, 특정 문서에 과적합된 쿼리 확장 문제를 발견했고, 이를 일반화 가능한 방식으로 개선했다.

`docs/initial-plan.md`의 예시 구현과 현재 `indexer.py`, `src/retrieval/hybrid.py`는 일부 다르다. 이는 설계를 무시한 것이 아니라, 실제 PDF 테스트 중 발견한 출처 오류, 페이지 번호 차이, 청킹 실패, reranker 호환성 문제를 반영해 더 안정적인 구현으로 확장한 결과다.

## 구체적 설명

### Phase 1: 프로젝트 초기 구조 구성

프로젝트 기본 폴더와 설정 파일을 구성했다.

주요 생성 파일과 폴더:

- `README.md`
- `requirements.txt`
- `config.py`
- `src/`
- `data/pdfs/`
- `chroma_db/`
- `indexes/`
- `tests/`
- `app.py`

Python 가상환경과 의존성 설치 절차를 README에 정리했다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ollama 모델은 명시 태그를 사용하도록 정리했다.

```bash
ollama pull exaone3.5:7.8b
```

### Phase 2: 기본 RAG 파이프라인 구현

PDF 로딩, 청킹, 임베딩, Chroma 저장, 단일 질의응답 흐름을 구현했다.

주요 구현 내용:

- `src/pdf_loader.py`
  - `pymupdf4llm` 기반 PDF Markdown 추출
  - 페이지별 `Document` 생성
  - ChromaDB에 안전한 scalar metadata만 저장
  - PDF에 인쇄된 페이지 번호를 우선 추출하도록 개선

- `src/indexer.py`
  - Markdown header 기반 1차 분할
  - recursive text splitter 기반 2차 분할
  - 페이지 전체 컨텍스트 청크 추가
  - `KURE-v1` 임베딩 생성
  - ChromaDB 인덱스 생성 및 재생성

- `src/retriever.py`
  - Chroma vector search
  - 검색 결과 context 포맷팅
  - metadata 기반 출처 페이지 계산

- `src/llm_client.py`
  - Ollama chat 호출
  - 스트리밍 응답 지원

- `src/main.py`
  - `index`, `ask`, `chat` CLI 명령 제공

기본 실행 예시는 다음과 같다.

```bash
python -m src.main index
python -m src.main --reuse-index ask "질문"
python -m src.main --reuse-index chat
```

### Phase 3: 대화형 기능 구현

멀티턴 대화, 스트리밍 응답, 시스템 프롬프트, 후속 질문 검색 보강을 구현했다.

주요 구현 내용:

- `ChatManager` 구현
  - `MAX_HISTORY_TURNS`를 대화 턴 단위로 관리
  - 현재 질문은 답변 생성 후 history에 추가
  - 짧은 후속 질문은 이전 사용자 질문과 결합해 retrieval query 보강

- 스트리밍 응답 개선
  - Ollama streaming 응답 출력
  - 모델이 생성한 `(p.12 참조)`, `(출처: ...)` 같은 inline citation 제거

- 시스템 프롬프트 개선
  - 문서에 없는 내용은 추측하지 않도록 지시
  - 답변 본문에 출처를 직접 쓰지 않고, 시스템이 별도로 붙이도록 변경
  - 조문 번호와 페이지 번호 혼동 방지

### Phase 4: 정확도 개선

검색 정확도를 높이기 위해 vector only 검색을 BM25 + vector hybrid retrieval로 확장했다.

주요 구현 내용:

- 한국어 BM25 검색
  - `kiwipiepy` 형태소 토크나이저 적용
  - `rank-bm25` 기반 BM25 검색 구현

- Hybrid retrieval
  - Chroma vector search 결과와 BM25 결과를 각각 정규화
  - `VECTOR_WEIGHT=0.6`, `BM25_WEIGHT=0.4`로 `hybrid_score` 계산
  - vector/BM25 중복 청크를 병합
  - `retrieval`, `vector_score`, `bm25_score`, `hybrid_score` metadata 추가

- BM25 청크 캐시
  - `indexes/chunks.json`에 청크 저장
  - `--reuse-index` 모드에서도 BM25 재구성 가능

- Reranker
  - `USE_RERANKER=true` 환경 변수로 선택 실행
  - 기존 `FlagEmbedding.FlagReranker`는 현재 환경에서 tokenizer API 호환성 문제가 있어 제거
  - `transformers.AutoModelForSequenceClassification` 기반 `CrossEncoderReranker`로 대체

- 청크 크기 튜닝
  - `tests/eval_cases.py` 평가셋 추가
  - `scripts/tune_chunks.py` 튜닝 스크립트 추가
  - `800/1200/1500/2000` 후보 비교
  - 최종 추천값 `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=120` 반영

## 결과

현재 가능한 기능:

- `data/pdfs/` 안의 PDF 전체 인덱싱
- PDF Markdown 추출
- 페이지 metadata 보존
- 한국어 임베딩 기반 벡터 검색
- 한국어 BM25 검색
- BM25 + vector hybrid retrieval
- 선택적 reranker 실행
- Ollama 기반 한국어 답변 생성
- 단일 질문 실행
- 대화형 질문 실행
- 멀티턴 후속 질문 처리
- 스트리밍 응답 출력
- metadata 기반 출처 페이지 표시
- PDF 내부 인쇄 페이지 번호 기반 출처 표시
- retrieval/source 평가 스크립트
- latency 측정 스크립트

검증 결과:

```bash
.venv/bin/python -m pytest
```

최종 테스트 결과:

```text
43 passed
```

블록체인 PDF 테스트에서 확인한 개선 결과:

```text
블록체인의 4가지 주요 기술적 특징:
1. 탈중앙성
2. 투명성
3. 불변성
4. 가용성

출처: p.22
```

문화 콘텐츠 관련 질문에서도 다음처럼 개선되었다.

```text
현재 온라인 문화 콘텐츠 창작자들은 주로 포털이나 플랫폼 사이트에 콘텐츠를 업로드하고 있습니다.
명시적으로 언급된 플랫폼 예시로는 유튜브와 아프리카TV가 있습니다.

출처: p.120, p.121
```

청크 크기 튜닝 결과:

```text
800   → page hit 4/5, term hit 5/5
1200  → page hit 5/5, term hit 5/5
1500  → page hit 5/5, term hit 5/5
2000  → page hit 5/5, term hit 5/5
```

최종 선택:

```python
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
```

## 트러블 슈팅 내용

### 1. 모델이 조문 번호를 페이지 번호로 착각

문제:

`제12조`를 모델이 `p.12`로 잘못 표기했다.

원인:

출처 표기를 모델에게 맡기고 있었기 때문에, 조문 번호와 페이지 번호를 혼동했다.

해결:

- 모델 프롬프트에서 답변 본문에 페이지 번호를 쓰지 말도록 지시
- 출처는 검색된 `Document.metadata["page"]`에서만 계산
- inline citation 제거 함수 추가

### 2. PDF 내부 페이지와 파일 페이지가 다름

문제:

보고서 PDF에서 사용자가 보는 페이지 번호와 PyMuPDF 기준 페이지 번호가 달랐다.

예: 파일 page 23에 인쇄된 번호는 `22`.

해결:

- PDF 본문 첫 줄 또는 마지막 줄에 있는 숫자 페이지를 우선 추출
- 출처 표시에는 인쇄된 페이지 번호를 우선 사용

### 3. 4가지 기술적 특징을 못 찾는 문제

문제:

문서 p.22에 블록체인의 4가지 기술적 특성이 모두 있었지만, 답변은 “명시적으로 찾을 수 없다”고 나왔다.

원인:

Markdown header splitter가 다음 항목들을 각각 다른 청크로 쪼갰다.

- `1. 탈중앙성`
- `2. 투명성`
- `3. 불변성`
- `4. 가용성`

모델 입장에서는 4가지가 한 번에 나열된 근거를 보기 어려웠다.

해결:

- 페이지 전체 컨텍스트 청크를 추가 저장
- 번호 목록이 한 페이지 안에 있을 때 전체 구조가 검색 context에 포함되도록 개선

### 4. 멀티턴 후속 질문에서 출처가 흔들림

문제:

첫 질문 후 `그 예외는?`처럼 짧은 후속 질문을 하면 검색 query가 너무 짧아져 출처가 흔들렸다.

해결:

- `ChatManager.build_retrieval_query()` 추가
- 최근 사용자 질문과 현재 질문을 결합해 retrieval query 생성

### 5. 특정 PDF에 과적합된 쿼리 확장 문제

문제:

문화 콘텐츠 질문을 해결하기 위해 처음에는 아래처럼 특정 단어 기반 보강을 추가했다.

```python
NAMED_SOURCE_TERMS = ["유튜브", "아프리카tv", "스팀잇"]
```

하지만 이 방식은 다른 책이나 다른 도메인에 일반화되지 않는다.

해결:

- 특정 단어 리스트 제거
- 도메인 특화 `QUERY_EXPANSIONS` 제거
- 대신 답변에 나온 구체 명칭과 검색 청크의 원문을 비교하는 일반 로직 추가
- `extract_named_terms()`로 한국어/영문 고유명사 후보를 자동 추출

### 6. FlagEmbedding reranker 호환성 문제

문제:

`USE_RERANKER=true` 실행 시 다음 오류가 발생했다.

```text
AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
```

원인:

현재 환경의 `transformers 5.8.1`과 `FlagEmbedding.FlagReranker` 내부 구현이 맞지 않았다. `FlagReranker`가 tokenizer의 구버전 API인 `prepare_for_model()`에 의존하고 있었다.

해결:

- `FlagEmbedding.FlagReranker` 직접 사용 제거
- `transformers.AutoTokenizer`
- `transformers.AutoModelForSequenceClassification`
- 위 조합으로 `CrossEncoderReranker` 직접 구현

검증:

`USE_RERANKER=true` 실행 시 `rerank_score`가 정상적으로 생성됨을 확인했다.

### 7. 초기 계획 대비 구현 차이

`docs/initial-plan.md`의 `indexer.py` 예시는 구조가 단순했다.

```text
PDF 로딩
→ Markdown header split
→ recursive split
→ Chroma 저장
→ BM25 메모리 인덱스 생성
```

현재 `indexer.py`는 여기에 다음을 추가했다.

- 페이지 전체 컨텍스트 청크 추가
- 청크 크기 실험용 `split_markdown_documents_with_config()` 추가
- 안정적 chunk id 생성
- BM25 재사용을 위한 `indexes/chunks.json` 저장/로드
- `--reuse-index`에서 Chroma와 BM25 모두 재사용

이 방식이 더 나은 이유:

- 번호 목록, 표, 한 페이지짜리 요약처럼 header split만으로 의미가 깨지는 경우를 보완한다.
- 재시작 후에도 BM25 검색을 유지할 수 있다.
- chunk size를 평가 기반으로 튜닝할 수 있다.

`docs/initial-plan.md`의 `retriever.py` 예시는 LangChain `EnsembleRetriever`와 `FlagReranker` 중심이었다.

현재 `retriever.py`는 직접 hybrid retrieval을 구현했다.

```text
vector 검색
BM25 검색
점수 정규화
weighted merge
optional reranker
출처 페이지 필터링
```

이 방식이 더 나은 이유:

- `vector_score`, `bm25_score`, `hybrid_score`, `rerank_score`를 직접 확인할 수 있다.
- BM25-only 후보가 출처에 과하게 섞이는 문제를 제어할 수 있다.
- 답변에 실제 사용된 고유명사와 청크 원문을 비교해 출처를 보강할 수 있다.
- `FlagReranker` 호환성 문제를 피할 수 있다.

단점:

- `retriever.py`가 무거워졌다.
- 추후 `bm25.py`, `reranker.py`, `source_resolver.py`로 분리하는 리팩터링이 필요하다.

### 8. 다중 PDF 법률 참조 추적 보강

문제:

`data/pdfs`에 `근로기준법`과 `근로자퇴직급여 보장법` PDF가 함께 있을 때, `근로기준법` 제43조의2에 있는 `「근로자퇴직급여 보장법」 제12조제1항` 참조를 읽기는 했지만, 실제 답변에서는 참조 대상 PDF의 제12조 원문까지 연결하지 못했다.

원인:

- hybrid 검색은 질문과 직접 유사한 `근로기준법` 청크를 우선 반환했다.
- 검색기가 조문 안에 포함된 다른 법률 참조를 별도 검색 대상으로 확장하지 않았다.
- reranker를 사용하면 상위 후보 수가 줄어들어 참조 대상 청크가 더 쉽게 탈락했다.

해결:

- `「법률명」 제N조제M항` 형식의 법률 참조 추출 추가
- 추출된 법률명과 조문 번호를 기준으로 전체 청크에서 참조 대상 조문 검색
- reranker 사용 시에도 법률 참조로 찾은 청크는 최종 context에 보존
- 같은 법률에 여러 참조가 있을 경우 `제N조제M항`처럼 더 구체적인 참조를 우선
- 다중 PDF 출처는 `파일명 p.N` 형식으로 표시
- 법령 우선순위 표현을 뒤집지 않도록 system prompt 보강
- 모델이 본문에 넣는 `(문서 참조: ...)` 형식의 인라인 참조 제거 추가

검증:

```bash
USE_RERANKER=true python -m src.main --reuse-index ask "체불사업주 명단공개는 무엇이고, 퇴직급여는 어떻게 되는가?"
```

결과:

- `근로기준법` 제43조의2 내용과 `근로자퇴직급여 보장법` 제12조 내용을 함께 답변
- 퇴직급여등이 질권 또는 저당권 담보채권을 제외하고 조세ㆍ공과금 및 다른 채권에 우선하여 변제된다고 설명
- 출처가 `근로기준법 p.6; 근로자퇴직급여 보장법 p.4`로 표시됨

## 완료 현황

완료된 작업:

- Phase 1 환경 구성
- Phase 2 기본 RAG 파이프라인
- Phase 3 대화형 기능
- Phase 4 정확도 개선
- 한국어 BM25 토크나이저 적용
- BM25 + vector hybrid retrieval
- 선택적 reranker 구현
- 청크 크기 튜닝
- 스트리밍 응답
- 출처 페이지 metadata 기반 표시
- 인쇄 페이지 번호 처리
- 페이지 전체 컨텍스트 청킹
- 멀티턴 검색 query 보강
- 특정 문서 과적합 규칙 제거
- 일반화 가능한 고유명사 출처 보강
- 다중 PDF 법률 참조 추적
- 다중 PDF 파일명 포함 출처 표시
- 다중 PDF 환경에서 단일 자료 출처도 파일명 포함 표시
- 실제 답변에 쓰이지 않은 법률 참조 문서는 최종 출처에서 제외
- 약한 후보의 법률 참조 과잉 확장 방지
- 검색 결과 디버그 모드 추가
- reranker 품질/속도 비교
- 잘못된 출처 회귀 테스트 추가
- 테스트 37개 통과

현재 실행 명령:

```bash
source .venv/bin/activate
python -m src.main index
python -m src.main --reuse-index ask "질문"
python -m src.main --reuse-index ask --debug-retrieval "질문"
python -m src.main --reuse-index chat
python -m src.main --reuse-index chat --debug-retrieval
```

### 9. retriever 모듈 분리 리팩터링

문제:

`src/retriever.py`가 805줄까지 비대해져 BM25, reranker, 법률 참조 추적, 출처 페이지 결정 로직이 한 파일에 섞여 있었다. 후속 작업(디버그 모드, 평가 추가)을 진행하기 전 모듈 경계를 정리할 필요가 있었다.

해결:

`retriever.py`를 hybrid retrieval orchestrator로만 남기고 책임별로 5개 모듈로 분리했다.

- `src/document_utils.py` — 공통 유틸 (`doc_key`, `clone_document`, `merge_unique_documents`, `normalize_scores`, `tokenize_for_evidence`, `token_overlap_ratio`, `TOKEN_PATTERN`, `LAW_ARTICLE_PATTERN`)
- `src/bm25.py` — `get_kiwi`, `tokenize_korean`, `retrieve_bm25`
- `src/reranker.py` — `CrossEncoderReranker`, `get_reranker`, `rerank_documents`
- `src/legal_references.py` — `extract_legal_references`, `legal_reference_match_score`, `retrieve_legal_references`, `reference_doc_used_in_answer`, `normalize_reference_text`
- `src/source_resolver.py` — `is_no_evidence_answer`, `format_source_name`, `extract_named_terms`, `score_evidence_doc`, `select_source_docs`, `select_answer_source_docs`, `get_source_pages`, `format_source_pages`, `IMPORTANT_TERMS`, `NO_EVIDENCE_PATTERNS`
- `src/retriever.py` — `expand_query`, `retrieve`, `merge_hybrid_results`, `retrieve_hybrid`, `format_context`

호출처(`src/main.py`, `scripts/tune_chunks.py`, `tests/test_cases.py`)도 새 모듈 경로에서 import 하도록 업데이트했다. retriever.py를 통한 재노출(re-export)은 추가하지 않아 canonical import 경로가 한 곳으로 유지된다.

`retrieve_hybrid()` 안에서 `expand_query()`를 한 번만 호출하도록 정리했다. 기존에는 `retrieve()`와 `retrieve_bm25()` 각자에서 호출했다.
검토 중 `retrieve()` 단독 사용을 위해 내부 expansion이 남아 있어 `retrieve_hybrid()` 경로에서 한 번 더 호출될 수 있음을 확인했고, `retrieve(..., expand=False)` 옵션과 회귀 테스트를 추가해 hybrid 경로에서는 한 번만 호출되도록 고정했다.

검증:

```bash
.venv/bin/python -m pytest
```

```text
31 passed
```

리팩터링 전후 라인 수:

```text
이전:  retriever.py 805줄
이후:  retriever.py 172줄
       source_resolver.py 326줄
       legal_references.py 216줄
       reranker.py 111줄
       document_utils.py 80줄
       bm25.py 74줄
       retrieval_debug.py 89줄
       (총 1047줄, +242줄은 모듈 헤더/import, 디버그 출력, 법률 참조 필터링 비용)
```

### 10. 검색 결과 디버그 모드

문제:

검색 품질 이슈를 확인할 때 최종 답변과 출처만으로는 어떤 단계에서 후보가 잘못 들어왔는지 구분하기 어려웠다. 특히 vector 검색, BM25 검색, 법률 참조 추적, reranker 적용 후 결과를 따로 확인할 방법이 필요했다.

해결:

`src/retrieval_debug.py`를 추가해 retrieval trace를 사람이 읽기 좋은 터미널 출력으로 포맷하도록 했다. `retrieve_hybrid()`는 선택적으로 `RetrievalDebugTrace`를 받아 다음 단계의 후보를 기록한다.

- vector candidates
- BM25 candidates
- merged candidates
- legal reference candidates
- reranked candidates
- final context docs

CLI에는 `--debug-retrieval` 옵션을 추가했다.

```bash
python -m src.main --reuse-index ask --debug-retrieval "질문"
python -m src.main --reuse-index chat --debug-retrieval
```

디버그 출력에는 문서명, 페이지, retrieval 종류, score/vector/bm25/hybrid/rerank/evidence 점수, 본문 preview가 표시된다.

검증:

```bash
.venv/bin/python -m pytest
```

```text
34 passed
```

### 11. 약한 후보의 법률 참조 과잉 확장 방지

문제:

`근로기준법 출석의 의무` 질문에서 최종 답변은 근로기준법 제13조만 사용했지만, 낮은 순위 BM25 후보에 포함된 `근로자퇴직급여 보장법 제2조` 참조가 확장되어 출처에 `근로자퇴직급여 보장법 p.1`이 함께 표시되었다.

해결:

`retrieve_legal_references()`가 후보 청크의 법률 참조를 따라갈 때 다음 조건을 추가했다.

- 질문과 후보 청크의 token overlap이 일정 기준 이상일 것
- 참조 조항이 `제N조제M항`처럼 구체적이거나, 참조 법률명 자체가 질문의 핵심어와 관련 있을 것

이로써 `출석의 의무`처럼 퇴직급여와 무관한 질문에서는 퇴직급여법 참조를 확장하지 않고, `체불사업주 명단공개와 퇴직급여`처럼 실제로 퇴직급여가 질문/답변에 관련된 경우에는 기존 교차 법률 추적을 유지한다.

검증:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m src.main --reuse-index ask "근로기준법 출석의 의무"
```

```text
34 passed
출처: 근로기준법 p.2
```

### 12. reranker 품질/속도 비교

목적:

`USE_RERANKER=true`가 실제로 답변 정확도를 높이는지, 속도 비용을 감당할 만한지 확인했다.

구현:

`scripts/compare_reranker.py`를 추가했다. 현재 `data/pdfs/`에 들어있는 법률 PDF 기준으로 5개 평가 질문을 사용해 reranker off/on을 같은 인덱스에서 비교한다.

측정 기준:

- page hit: 기대 문서명과 페이지가 최종 검색 결과에 포함되는지
- term hit: 기대 핵심어가 최종 컨텍스트에 포함되는지
- both hit: page hit와 term hit가 모두 충족되는지
- latency: 검색 1회당 소요 시간

실행:

```bash
.venv/bin/python scripts/compare_reranker.py --repeat 2
```

결과:

```text
| mode | page hit | term hit | both hit | avg | median | first | avg after first |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 5/5 | 4/5 | 4/5 | 0.81s | 0.54s | 3.26s | 0.54s |
| on  | 4/5 | 4/5 | 3/5 | 5.20s | 4.75s | 10.41s | 4.62s |
```

해석:

현재 법률 PDF 평가셋에서는 reranker가 검색 품질을 개선하지 못했다. 오히려 `체불사업주 명단공개는 무엇인가?` 케이스에서 기대 페이지 일부를 최종 context에서 밀어내 page hit가 떨어졌다. 속도도 warm 상태 기준 약 0.54초에서 4.62초로 약 8.6배 느려졌다.

결론:

현재 기본값은 `USE_RERANKER=false`를 유지한다. reranker는 지금 단계에서는 상시 활성화하지 않고, 향후 모델 교체나 top_k 조정 후 다시 비교한다.

### 13. 잘못된 출처 회귀 테스트 추가

목적:

Phase 5에서 발견한 잘못된 출처 표시 케이스를 회귀 테스트로 고정했다.

추가한 케이스:

- 블록체인 질문인데 현재 코퍼스가 법률 PDF뿐이면 `출처: 관련 문서 없음`으로 표시
- 답변이 한 문서 내용만 사용했으면 검색 결과에 다른 PDF가 있어도 최종 출처에서 제외
- 답변 없음 문장에서는 여러 PDF가 검색되더라도 출처를 붙이지 않음

테스트 추가 중 확인된 문제:

- `관련 내용을 확인할 수 없습니다` 형태의 no-evidence 문장을 감지하지 못했다.
- 답변의 `근로자`가 다른 문서의 `근로자퇴직급여`에 부분 문자열로 매칭되어 다른 PDF가 출처에 붙을 수 있었다.

해결:

- no-evidence 패턴에 `관련 내용/정보를 확인할 수 없습니다` 계열 표현을 추가했다.
- 답변 고유명사 매칭을 단순 부분 문자열 검색에서 조사 제거 후 token 단위 비교로 변경했다.

검증:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m src.main --reuse-index ask "블록체인 유형별 특징"
.venv/bin/python -m src.main --reuse-index ask "근로기준법 출석의 의무"
```

```text
37 passed
출처: 관련 문서 없음
출처: 근로기준법 p.2
```

### 14. Phase 6 Chainlit UI 시작

목적:

CLI로만 사용하던 RAG 챗봇을 브라우저 기반 채팅 UI에서 사용할 수 있게 한다.

구현:

- `app.py`에 Chainlit entry point 구현
- `src/rag_service.py` 추가
  - PDF 목록 수집
  - Chroma vectorstore와 chunk cache 로딩
  - 인덱스가 없을 때 자동 생성 옵션
  - 다중 PDF 여부 판단
  - 검색 context와 LLM messages 준비
  - 답변 완료 후 metadata 기반 출처 포맷팅
- Chainlit session에 `ChatManager`와 RAG resources 저장
- 사용자 질문마다 기존 Phase 3 대화 history를 사용해 retrieval query 보강
- Ollama streaming 응답을 Chainlit 메시지로 스트리밍 표시
- 답변 마지막에 `format_source_pages()` 결과를 붙여 CLI와 동일한 출처 정책 유지

실행:

```bash
chainlit run app.py
```

기본 동작:

- 기존 인덱스가 있으면 재사용
- 인덱스가 없으면 `data/pdfs/`의 PDF로 자동 생성
- 인덱스를 강제로 다시 만들려면 `CHAINLIT_REBUILD_INDEX=true` 사용

검증:

```bash
.venv/bin/python -m py_compile app.py src/rag_service.py src/main.py
.venv/bin/python -m pytest
```

```text
37 passed
```

### 15. PDF 업로드 및 source 표시 UI 개선

목적:

Chainlit UI에서 PDF를 추가하고, 답변 출처를 단순 텍스트 한 줄보다 확인하기 쉬운 UI로 개선한다.

구현:

- `/upload` 명령 추가
  - `cl.AskFileMessage`로 PDF 업로드 창 표시
  - 최대 10개, 파일당 100MB까지 PDF 업로드 허용
  - 업로드 파일은 `data/pdfs/`에 저장
  - 같은 파일명이 있으면 `_2`, `_3` suffix를 붙여 기존 PDF를 덮어쓰지 않음
- 업로드 후 인덱스 재생성
  - 전체 `data/pdfs/` 기준으로 Chroma와 chunk cache 재생성
  - 문서 묶음 변경 후 이전 대화 맥락이 섞이지 않도록 UI session history 초기화
- 첨부 PDF 처리
  - 메시지에 PDF element가 함께 들어오면 `/upload` 명령 없이도 저장 후 재색인 가능
- source 표시 UI 개선
  - 답변 마지막의 metadata 기반 출처 표시는 유지
  - `build_source_references()`로 실제 선택된 source doc의 문서명, 페이지, preview 생성
  - Chainlit side panel에 `출처 근거` Text element로 근거 청크 표시

후속 변경:

- Chainlit side panel의 `출처 근거` preview는 제거했다.
- 현재 UI는 답변 본문 마지막에 `출처: 문서명 p.N` 형식만 표시한다.

검증:

```bash
.venv/bin/python -m py_compile app.py src/rag_service.py tests/test_cases.py
.venv/bin/python -m pytest
```

```text
39 passed
```

### Phase 6 완료 정리

Phase 6 UI 범위는 완료로 정리한다.

완료 항목:

- Chainlit 기본 채팅 UI
- UI 대화 history 연결
- CLI와 동일한 검색/출처 표시 정책 재사용
- PDF 업로드 기능
- 답변 source 본문 표시

현재 UI 동작:

- Chainlit에서 질문/답변을 streaming으로 주고받을 수 있다.
- `/upload` 또는 PDF 첨부로 파일을 `data/pdfs/`에 저장하고 전체 인덱스를 재생성한다.
- PDF 변경 후에는 이전 대화 history를 초기화해 문서 맥락이 섞이지 않게 한다.
- 답변 마지막에는 metadata 기반 출처가 `문서명 p.N` 형식으로 표시된다.

### 16. Phase 7: 평가/Latency 스크립트 정리

목적:

테스트셋 확장은 뒤로 미루고, 현재 RAG 품질을 반복 측정할 수 있는 실행 도구부터 정리한다. 특히 검색 정확도, preliminary source 선택 품질, latency 병목을 분리해서 확인할 수 있게 한다.

구현:

- `src/evaluation.py` 추가
  - 법률/블록체인 평가 케이스 공통 정의
  - 기대 source/page hit 계산
  - 기대 핵심어 hit 계산
  - 선택된 출처의 extra ref 계산
  - latency 평균/median/p95 계산
- `scripts/evaluate_retrieval.py` 추가
  - LLM 호출 없이 retrieval 정확도와 preliminary source 선택 품질 측정
  - `--case-set all|legal|blockchain`
  - `--repeat`
  - `--reranker env|on|off`
  - `--json-out`
- `scripts/measure_latency.py` 추가
  - resource load, retrieval, prompt build, source formatting, optional LLM generation 시간 분리 측정
  - 기본값은 LLM 미포함
  - `--include-llm`을 주면 Ollama 생성 시간까지 측정
- 기존 `scripts/compare_reranker.py`와 `scripts/tune_chunks.py`가 새 공통 평가 유틸을 사용하도록 정리
- `tests/eval_cases.py`는 기존 import 호환용 wrapper로 변경

검증:

```bash
.venv/bin/python -m py_compile src/evaluation.py scripts/evaluate_retrieval.py scripts/measure_latency.py scripts/compare_reranker.py scripts/tune_chunks.py
.venv/bin/python -m pytest
.venv/bin/python scripts/evaluate_retrieval.py --case-set legal --repeat 1
.venv/bin/python scripts/measure_latency.py --case-set legal --limit 2 --repeat 1 --warmup 0
```

결과:

```text
43 passed
```

Retrieval/source 평가 결과:

```text
expected source/page: 6/6
selected source hit: 2/6
selected source clean: 4/6
expected terms: 5/6
both: 5/6
avg retrieval latency: 2.22s
```

확인된 개선 후보:

- `cross_law_mention`은 preliminary source에 `근로기준법 p.1`이 불필요하게 섞인다.
- `retirement_priority`는 preliminary source에 `근로기준법 p.6`이 불필요하게 섞인다.
- `labor_standard_scope_exception`은 source는 맞지만 기대 핵심어 중 일부가 검색 context 평가에서 빠진다.

Latency 측정 결과:

```text
resource load: 5.95s
retrieval avg: 3.78s
retrieval median: 3.78s
retrieval p95: 5.51s
LLM generation: not measured
```

### 17. Phase 7: retrieval 패키지 분리 리팩터링

목적:

`src/` 루트에 retrieval 관련 파일이 많아져 책임 경계를 한눈에 보기 어려웠다. 정확도 개선과 출처 noise 개선 작업을 계속 진행하기 전에 retrieval 계열만 별도 패키지로 묶었다.

변경:

- `src/retrieval/` 패키지 생성
- retrieval 관련 파일 이동
  - `src/retriever.py` → `src/retrieval/hybrid.py`
  - `src/bm25.py` → `src/retrieval/bm25.py`
  - `src/reranker.py` → `src/retrieval/reranker.py`
  - `src/legal_references.py` → `src/retrieval/legal_references.py`
  - `src/source_resolver.py` → `src/retrieval/source_resolver.py`
  - `src/retrieval_debug.py` → `src/retrieval/debug.py`
  - `src/document_utils.py` → `src/retrieval/document_utils.py`
- `src/main.py`, `src/rag_service.py`, `src/evaluation.py`, scripts, tests의 import 경로를 새 canonical path로 수정
- compatibility wrapper는 만들지 않고 새 경로를 기준으로 정리

검증:

```bash
.venv/bin/python -m py_compile src/main.py src/rag_service.py src/evaluation.py src/retrieval/*.py scripts/evaluate_retrieval.py scripts/measure_latency.py scripts/compare_reranker.py scripts/tune_chunks.py tests/test_cases.py
.venv/bin/python -m pytest
.venv/bin/python scripts/evaluate_retrieval.py --help
.venv/bin/python scripts/measure_latency.py --help
```

결과:

```text
43 passed
```

## 다음 예정 작업

### Phase 7: 평가/성능 측정 및 구조 정리

완료:
- retrieval/source 평가 스크립트
- latency 측정 스크립트
- retrieval 패키지 분리
- Phase 6 UI 완료 정리

예정 작업:
- preliminary source noise 개선
- 복합 질문 answer-aware source 평가 보강
- LLM 포함 latency 측정
- 평가용 Q&A 테스트셋 작성
- 인덱스 증분 업데이트
- 파일 hash 기반 중복 인덱싱 방지
- Chroma collection 관리 명령 추가

### 추후 재확인 작업

- debug 기능
