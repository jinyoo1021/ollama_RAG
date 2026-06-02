# Ollama 기반 PDF RAG 챗봇 초기 계획서

> 이 문서는 프로젝트 초기에 작성한 조사 및 설계 계획입니다.
> 현재 구현 상태와 실행 방법은 [README.md](../README.md)를 기준으로 확인하고,
> 개발 과정의 변경 이유와 트러블슈팅은 [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md)를 참고하세요.

---

## 1. 프로젝트 개요

### 1.1 목적

로컬 환경에서 동작하는 PDF 문서 기반 한국어 질의응답 챗봇 구축. 외부 API에 의존하지 않아 **보안성**과 **비용 효율성** 확보.

### 1.2 핵심 기능

- PDF 문서 업로드 및 자동 인덱싱
- 자연어 질문 → 문서 기반 정확한 답변
- 멀티턴 대화(대화 히스토리 유지)
- 답변 출처(페이지) 표시
- 스트리밍 응답

### 1.3 비기능 요구사항

- 응답 속도: 쿼리 임베딩 GPU/MPS 기준 < 1초, CPU 기준은 실측 후 조정, 답변 생성 < 30초 (7B 모델 기준)
- 정확도: 문서 내 명시된 사실에 대한 답변 정확도 90% 이상
- 환각(Hallucination) 최소화: 문서에 없으면 "찾을 수 없음" 명시

---

## 2. 기술 스택

### 2.1 두 블로그 vs 추가 옵션 비교 (전체 컴포넌트별)

각 컴포넌트마다 두 블로그의 선택, 그리고 제가 추가로 조사한 옵션들을 함께 비교하고 최종 추천을 정리했습니다.

#### 📄 PDF 파싱


| 도구                   | 속도        | 표 처리  | 한국어              | 라이선스       | 신·기록 | Judy | **추천**    |
| -------------------- | --------- | ----- | ---------------- | ---------- | ---- | ---- | --------- |
| **PyPDF / PyPDF2**   | 매우 빠름     | ❌     | 보통               | 자유         | -    | -    | ❌         |
| **PyMuPDFLoader**    | 빠름        | ⚠️ 기본 | 보통               | AGPL       | -    | ✅    | △         |
| **pymupdf4llm** ⭐    | 빠름        | ⭐⭐⭐   | 우수               | AGPL       | ✅    | -    | **✅ 1순위** |
| **Docling** (IBM)    | 느림 (4초/p) | ⭐⭐⭐⭐⭐ | 우수               | MIT        | -    | -    | ✅ 표 많은 문서 |
| **Marker**           | 중간 (8초/p) | ⭐⭐⭐⭐  | 보통               | GPL        | -    | -    | △ 학술 PDF  |
| **MinerU**           | 중간        | ⭐⭐⭐⭐  | **⭐⭐⭐⭐⭐ CJK 특화** | AGPL       | -    | -    | ✅ 복잡한 한국어 |
| **Unstructured**     | 느림        | ⭐⭐⭐⭐  | 우수               | Apache 2.0 | -    | -    | △         |
| **LlamaParse** (API) | 빠름        | ⭐⭐⭐⭐⭐ | 우수               | 유료         | -    | -    | ❌ 로컬 위배   |


> PDF 파싱 도구 비교에서, PyMuPDF4LLM은 텍스트 위주 PDF에 빠르고 가볍게 동작하며, Docling은 자가 호스팅이 필요하고 레이아웃 인식이 필요할 때 좋고, Marker는 학술 논문이나 책 같이 참조와 구조가 중요한 경우에 추천됩니다.
>
> 2026년 PDF-to-Markdown 도구 가이드에 따르면, 복잡한 학술 논문의 한중일 텍스트(CJK)에는 MinerU가 다른 어떤 도구보다 한국어/중국어/일본어 레이아웃 감지에 뛰어나며, 네이티브 PDF 빠른 추출에는 PyMuPDF4LLM, 종합형 용도로는 Marker가 추천됩니다.

**🎯 내 추천**: 일반 문서는 **pymupdf4llm**(빠르고 충분), 표·수식·복잡 레이아웃이 많으면 **MinerU**(한국어 특화), 학술 PDF면 **Marker**. 처음에는 pymupdf4llm로 시작해서 결과 보고 업그레이드하는 게 효율적.

---

#### ✂️ 청킹(Chunking) 전략


| 방식                                 | 장점         | 단점         | 신·기록         | Judy       | **추천**             |
| ---------------------------------- | ---------- | ---------- | ------------ | ---------- | ------------------ |
| **RecursiveCharacterTextSplitter** | 빠름, 안정적    | 의미 무시      | ✅ (1500/150) | ✅ (500/50) | **✅ 기본**           |
| **MarkdownHeaderTextSplitter**     | 마크다운 구조 보존 | MD 변환 필수   | -            | -          | ✅ pymupdf4llm 조합 시 |
| **SemanticChunker**                | 의미 단위로 분할  | 느림, 임베딩 비용 | -            | -          | △ 품질 우선 시          |
| **TokenTextSplitter**              | 토큰 정확 계산   | 의미 무시      | -            | -          | △ 한국어 토크나이저 주의     |
| **Late Chunking**                  | 컨텍스트 전체 보존 | 최신, 셋업 복잡  | -            | -          | 🆕 실험적             |


**🎯 내 추천**: `**MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` 2단계 적용**. pymupdf4llm가 만든 마크다운의 헤더(`#`, `##`)로 1차 분할 → 너무 큰 청크만 2차로 글자 단위 분할. 이렇게 하면 "장(章) 단위 컨텍스트"가 보존됩니다.

---

#### 🧮 임베딩 모델


| 모델                                  | 한국어       | 다국어   | 차원   | 비고                     | 신·기록 | Judy | **추천**           |
| ----------------------------------- | --------- | ----- | ---- | ---------------------- | ---- | ---- | ---------------- |
| `multilingual-e5-small`             | ⭐⭐⭐       | ⭐⭐⭐⭐  | 384  | 가벼움                    | ✅    | -    | △ 저사양            |
| `multilingual-e5-large`             | ⭐⭐⭐⭐      | ⭐⭐⭐⭐  | 1024 | 균형                     | -    | -    | ✅                |
| `BAAI/bge-m3` ⭐                     | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐ | 1024 | dense+sparse+multi-vec | -    | ✅    | ✅ 범용 1순위         |
| `**nlpai-lab/KURE-v1`** 🔥          | **⭐⭐⭐⭐⭐** | ⭐⭐    | 1024 | 고려대 NLP&AI Lab, 한국어 특화 | -    | -    | **✅ 한국어 PDF 최강** |
| `jhgan/ko-sroberta-multitask`       | ⭐⭐⭐⭐      | ⭐     | 768  | 한국어 전용 (오래됨)           | -    | -    | △                |
| `Alibaba-NLP/gte-multilingual-base` | ⭐⭐⭐⭐      | ⭐⭐⭐⭐  | 768  | 효율적                    | -    | -    | ✅ 대안             |
| `Qwen/Qwen3-Embedding-8B`           | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐ | 가변   | MTEB 다국어 1위            | -    | -    | ✅ GPU 좋으면        |
| Solar Embedding (Upstage)           | ⭐⭐⭐⭐⭐     | ⭐⭐⭐   | -    | API 유료                 | -    | -    | ❌ 로컬 X           |
| OpenAI text-embedding-3             | ⭐⭐⭐⭐      | ⭐⭐⭐⭐  | 가변   | API, 외부 전송             | -    | -    | ❌ 보안 X           |


> KURE-v1은 한국어 텍스트 검색에서 대부분의 다국어 임베딩 모델을 능가하는 뛰어난 성능을 보이며, 공개된 한국어 검색 모델 중 최고 수준의 하나로 알려져 있습니다.
>
> KURE 연구에 따르면, 기존 다국어 임베딩 모델들은 한국어의 고유한 언어적 특성을 온전히 반영하지 못하거나 장문 검색에서 성능 저하를 보이는 한계가 있는데, KURE는 한국어 검색 벤치마크에서 강력한 다국어 및 상용 모델들을 상회하는 성능을 보이며 특히 장문 검색에서 높은 경쟁력을 입증했습니다.
>
> 2025년 RAG용 임베딩 모델 비교에서, BGE-M3는 단일 모델로 dense·sparse·multi-vector 검색을 모두 처리하는 패러다임 전환을 보여줬으며, Qwen3-Embedding-8B는 2025년 6월 기준 MTEB 다국어 리더보드 챔피언(70.58점)으로 사용자 정의 임베딩 차원(32~4096)을 지원합니다.

**🎯 내 추천 (한국어 PDF RAG 전용 최강 조합)**:

- **1순위 `KURE-v1`** (한국어 PDF만 다룬다면 최고). 고려대학교 연구실 산출물로 한국어 검색에 최적화됨
- **2순위 `bge-m3`** (한+영 혼용, 다국어 PDF, 일반 용도)
- **3순위 `gte-multilingual-base`** (가볍게 가고 싶을 때)

> 두 블로그에서 사용한 `multilingual-e5-small`과 `bge-m3`보다 한 단계 위 선택지가 있다는 게 핵심 차이점입니다.

---

#### 🗄️ 벡터 데이터베이스


| DB             | 셋업            | 영속성       | 메타데이터 필터 | 스케일  | 신·기록 | Judy | **추천**         |
| -------------- | ------------- | --------- | -------- | ---- | ---- | ---- | -------------- |
| **ChromaDB**   | 매우 쉬움         | ✅         | ⭐⭐⭐      | 소규모  | ✅    | ✅    | **✅ 1순위 (개인)** |
| **FAISS**      | 쉬움            | △ (수동 저장) | ❌        | 중간   | -    | -    | △ 속도 우선        |
| **LanceDB** 🆕 | 쉬움            | ✅         | ⭐⭐⭐⭐     | 중간   | -    | -    | ✅ 모던 대안        |
| **Qdrant**     | 중간            | ✅         | ⭐⭐⭐⭐⭐    | 대규모  | -    | -    | ✅ 상용 진입        |
| **Milvus**     | 어려움           | ✅         | ⭐⭐⭐⭐⭐    | 초대규모 | -    | -    | ❌ 오버킬          |
| **pgvector**   | PostgreSQL 필요 | ✅         | ⭐⭐⭐⭐⭐    | 중간   | -    | -    | ✅ 기존 DB 있을 때   |
| **Weaviate**   | 중간            | ✅         | ⭐⭐⭐⭐     | 대규모  | -    | -    | △              |


**🎯 내 추천**: **개인 프로젝트는 ChromaDB 유지**, 상용/팀 프로젝트로 가면 **Qdrant**(Docker 한 줄로 띄움, 메타데이터 필터링 강함). PostgreSQL 이미 쓰고 있다면 **pgvector**가 가장 합리적.

---

#### 🔎 검색(Retrieval) 전략


| 전략                            | 정확도 향상      | 복잡도   | 신·기록 | Judy | **추천**    |
| ----------------------------- | ----------- | ----- | ---- | ---- | --------- |
| **Vector Only**               | 기본          | ⭐     | ✅    | ✅    | 시작점       |
| **BM25 (키워드)**                | +5~10%      | ⭐     | -    | -    | 보조        |
| **Hybrid (Vector + BM25)**    | **+10~20%** | ⭐⭐    | -    | -    | **✅ 필수**  |
| **Reranking** (Cross-Encoder) | **+15~25%** | ⭐⭐    | -    | -    | **✅ 필수**  |
| **HyDE** (가상 답변으로 검색)         | +5~15%      | ⭐⭐⭐   | -    | -    | ✅ 추상 질문 多 |
| **Multi-Query** (질문 재작성)      | +5~10%      | ⭐⭐⭐   | -    | -    | △         |
| **Parent Document**           | +10%        | ⭐⭐    | -    | -    | ✅ 긴 문서    |
| **GraphRAG / LightRAG**       | +20% (관계형)  | ⭐⭐⭐⭐⭐ | -    | -    | 🆕 고급     |


**🎯 내 추천 (단계적 적용)**:

1. **MVP**: Vector Only (Judy 방식)
2. **개선 1차**: + BM25 (Hybrid) — `EnsembleRetriever` (가중치 0.4:0.6)
3. **개선 2차**: + Reranker (`bge-reranker-v2-m3`) — Top 10 검색 → Top 3 재정렬
4. **고급**: 질문 유형에 따라 HyDE나 Multi-Query 추가

> 두 블로그 모두 **단순 벡터 검색만 사용** — 정확도를 한 단계 끌어올리려면 Hybrid + Reranker가 가장 효율 좋은 업그레이드입니다.

---

#### 🔗 프레임워크


| 프레임워크                                    | 학습 곡선 | RAG 특화 | 생태계  | 신·기록   | Judy | **추천**        |
| ---------------------------------------- | ----- | ------ | ---- | ------ | ---- | ------------- |
| **LangChain Components + ollama-python** | 낮음~중간 | ⭐⭐⭐⭐   | 큼    | ✅      | ✅    | **✅ MVP 1순위** |
| **LangChain LCEL**                       | 중간    | ⭐⭐⭐⭐   | 가장 큼 | △ (부분) | ✅    | ✅ 확장 단계       |
| **LlamaIndex**                           | 낮음    | ⭐⭐⭐⭐⭐  | 큼    | -      | -    | ✅ RAG 전용      |
| **Haystack** (Deepset)                   | 중간    | ⭐⭐⭐⭐   | 중간   | -      | -    | △             |
| **Direct (ollama-python)**               | 낮음    | ⭐⭐     | -    | ✅      | -    | △ 아주 단순한 경우   |
| **DSPy**                                 | 높음    | ⭐⭐⭐⭐   | 작음   | -      | -    | 🆕 실험         |


**🎯 내 추천**: **MVP는 LangChain 컴포넌트 + ollama-python 직접 호출**. 문서 로딩·청킹·Chroma·BM25는 LangChain 생태계를 쓰되, 답변 생성과 스트리밍은 `ollama-python`으로 단순하게 시작합니다. Chainlit UI, 평가, API 서버까지 커지면 `langchain-ollama`의 `ChatOllama`와 LCEL 체인으로 옮기는 방식이 가장 안전합니다.

---

#### 📊 평가(Evaluation) 도구 (두 블로그에는 없는 영역)


| 도구           | 용도                            | 한국어               | **추천**   |
| ------------ | ----------------------------- | ----------------- | -------- |
| **RAGAS**    | Faithfulness, Relevancy 자동 측정 | △ (평가용 LLM 지정 필요) | ✅ 표준     |
| **DeepEval** | pytest 스타일 평가                 | △                 | ✅ CI 통합  |
| **TruLens**  | RAG 트레이싱·평가                   | △                 | △        |
| **자체 테스트셋**  | 도메인 특화 정확도                    | ⭐⭐⭐⭐⭐             | **✅ 필수** |


**🎯 내 추천**: **자체 테스트셋 20~50문항 + RAGAS 병행**. 자동 평가만 믿지 말고 사람이 작성한 정답을 기준으로 회귀 테스트하세요.

> ⚠️ **RAGAS 한국어 사용 시 주의**: RAGAS는 내부적으로 LLM을 호출해 평가 점수를 산출하는데, 기본 설정은 OpenAI 영어 모델을 가정합니다. 한국어 정답·답변에 그대로 적용하면 점수 신뢰도가 떨어지므로, 평가용 LLM을 한국어가 가능한 모델로 명시적으로 지정해야 합니다. 외부 API 독립이 목표라면 `ChatOllama(model="exaone3.5:7.8b")`를 우선 사용하고, GPT-4o/Claude는 선택 평가용으로만 둡니다.

---

#### 🖥️ UI 프레임워크


| 프레임워크               | 특징             | 신·기록 | Judy | **추천**     |
| ------------------- | -------------- | ---- | ---- | ---------- |
| **CLI**             | 의존성 없음         | ✅    | ✅    | 개발 단계      |
| **Streamlit**       | 빠른 프로토타입       | -    | -    | ✅ 대안       |
| **Gradio**          | ML 데모 표준       | -    | -    | △ 데모용      |
| **Chainlit**        | 챗봇 특화, 스트리밍 UI | -    | -    | **✅ 1순위**  |
| **Open WebUI**      | Ollama 호환 풀스택  | -    | -    | ✅ 별도 셋업 없이 |
| **FastAPI + React** | 본격 서비스         | -    | -    | △ 후순위      |


**🎯 내 추천**: **Chainlit** (대화형 RAG에 최적, 스트리밍·소스 표시 내장) > **Streamlit**(범용). 빠르게 시연만 보여줄 거면 **Open WebUI**(별도 코드 없이 RAG 가능).

---

### 2.2 최종 추천 스택 (Korean PDF RAG 전용)

두 블로그의 선택을 베이스로, 위 조사 결과를 반영해 정리한 **2026년 5월 기준 최적 조합**:

```
┌───────────────────────────────────────────────────────┐
│  PDF 파싱      pymupdf4llm (마크다운)                 │
│  ↓                                                    │
│  청킹          MarkdownHeader → Recursive 2단계       │
│  ↓                                                    │
│  임베딩        KURE-v1 (한국어) / bge-m3 (혼용)       │
│  ↓                                                    │
│  벡터 DB       ChromaDB (개인) / Qdrant (상용)        │
│  ↓                                                    │
│  검색          Hybrid (BM25 + Vector) Top 10          │
│  ↓                                                    │
│  리랭킹        bge-reranker-v2-m3 → Top 3             │
│  ↓                                                    │
│  LLM           exaone3.5:7.8b (Ollama)                │
│  ↓                                                    │
│  프레임워크    LangChain Components + ollama-python   │
│  ↓                                                    │
│  UI            Chainlit (or Streamlit)                │
│  ↓                                                    │
│  평가          자체 테스트셋 + RAGAS                  │
└───────────────────────────────────────────────────────┘
```

#### 구현 기준으로 확정할 결정

계획서의 추천과 코드 예시가 서로 다르게 움직이지 않도록, MVP 기준은 아래처럼 고정합니다.


| 항목     | MVP 결정                                                          | 이유                          |
| ------ | --------------------------------------------------------------- | --------------------------- |
| 임베딩    | `nlpai-lab/KURE-v1`                                             | 한국어 PDF 전용 검색 품질 우선         |
| 대안 임베딩 | `BAAI/bge-m3`                                                   | 한·영·중 등 다국어 PDF가 많을 때       |
| 청킹     | `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` | PDF의 장/절 구조 보존 후 큰 덩어리만 재분할 |
| 검색     | 1차는 Vector Only, 2차에서 Hybrid + Reranker                         | MVP 복잡도 제어, 이후 정확도 개선       |
| BM25   | 한국어 형태소 토크나이저 필수                                                | 기본 공백 분리는 한국어 검색 품질이 불안정    |
| Chroma | `langchain_chroma.Chroma`                                       | 최신 LangChain 권장 import      |
| 프레임워크  | LangChain 컴포넌트 + `ollama-python`                                | MVP에서는 단순성, 확장 시 LCEL 전환    |
| LLM    | `exaone3.5:7.8b`                                                | 한국어 품질과 로컬 실행성 균형           |
| UI     | Chainlit 우선                                                     | 스트리밍·출처 표시가 챗봇 UX와 잘 맞음     |


### 2.3 두 블로그 대비 핵심 업그레이드 포인트


| 영역  | 두 블로그 수준          | 내 추천 업그레이드            | 기대 효과         |
| --- | ----------------- | --------------------- | ------------- |
| 임베딩 | bge-m3 / e5-small | **KURE-v1**           | 한국어 검색 정확도 ↑↑ |
| 청킹  | 단순 글자 분할          | **마크다운 헤더 + 글자 2단계**  | 컨텍스트 보존 ↑     |
| 검색  | Vector only       | **Hybrid + Reranker** | 정확도 +15~25%   |
| 평가  | 없음 (눈으로 확인)       | **자체 테스트셋 + RAGAS**   | 회귀 방지, 객관화    |
| UI  | CLI               | **Chainlit**          | 출처·스트리밍 UX    |


### 2.4 한국어 LLM 모델 비교 및 추천 ⭐

#### 한국어 LLM 리더보드 참고 (BenchLM, 2026.05 기준)


| 순위  | 모델                 | 제공 형태           | 컨텍스트     | 한국어 평균 점수 | Ollama 사용         |
| --- | ------------------ | --------------- | -------- | --------- | ----------------- |
| 1   | Solar Pro 2        | Proprietary     | 128K     | 80.1      | ❌ (API only)      |
| 2   | HyperClova X       | Open Weight     | 128K     | 78.4      | △ (GGUF 변환 필요)    |
| 3   | A.X (SKT)          | Proprietary     | 64K      | 78.0      | ❌                 |
| 4   | K-Exaone           | Proprietary     | 256K     | 76.0      | ❌                 |
| 5   | **Exaone 4.0 32B** | **Open Weight** | **128K** | **75.2**  | **△ 공식 태그 확인 필요** |


> 상위권에는 한국 기업(LG, Naver, SKT, Upstage)의 한국어 특화 모델이 포진. 그중 **로컬에서 바로 쓸 수 있는 오픈 모델은 Exaone 시리즈가 사실상 최선**입니다.

#### Ollama에서 바로 쓰거나 확인 후 가져올 주요 모델


| 모델                 | 크기                    | 한국어   | 컨텍스트  | 특징                          | 추천 용도                  |
| ------------------ | --------------------- | ----- | ----- | --------------------------- | ---------------------- |
| **exaone3.5**      | 2.4B / 7.8B / **32B** | ⭐⭐⭐⭐⭐ | 32K   | LG AI, 한국어/영어 이중언어 특화       | **PDF RAG 1순위**        |
| **exaone4.0**      | 32B                   | ⭐⭐⭐⭐⭐ | 128K  | 공식 Ollama 태그 확인 필요          | 긴 PDF 처리 후보            |
| **qwen3.5**        | 0.8B ~ 122B           | ⭐⭐⭐⭐  | 128K+ | 알리바바, MoE 옵션                | 균형형, 추론 강함             |
| **qwen3.6**        | 27B / 35B             | ⭐⭐⭐⭐  | 128K  | 최신, 코딩·에이전트 강화              | 고급 사용자                 |
| **gemma4**         | e2b/e4b/26b/31b       | ⭐⭐⭐   | 128K  | 구글, 140+개 언어                | 다국어 혼용 문서              |
| **deepseek-r1**    | 8B / 32B / 70B        | ⭐⭐⭐   | 32K~  | 추론 특화                       | 분석형 질의                 |
| **EEVE-Korean**    | 10.8B                 | ⭐⭐⭐⭐  | 4K    | yanolja, Llama2 기반 한국어 파인튜닝 | 일반 한국어 대화 (컨텍스트 짧음 주의) |
| **Bllossom Llama** | 3B / 8B               | ⭐⭐⭐   | 8K    | 과기대 연구실, Llama3 기반          | 경량 한국어                 |


#### 하드웨어별 모델 선택 가이드


| 환경        | RAM/VRAM      | 추천 모델                                | 비고                                |
| --------- | ------------- | ------------------------------------ | --------------------------------- |
| 노트북 (CPU) | 8GB           | `exaone3.5:2.4b`                     | 의외로 한국어 잘함, 응답 10~20초             |
| 입문 GPU    | 8GB VRAM      | `exaone3.5:7.8b` (Q4_K_M)            | **가장 추천**, 안정적                    |
| 일반 GPU    | 12GB VRAM     | `exaone3.5:7.8b` (Q8) / `qwen3.5:9b` | 품질 향상                             |
| 고성능 GPU   | 16GB VRAM     | `exaone3.5:7.8b` (FP16)              | 양자화 없이 최고 품질                      |
| 워크스테이션    | 24GB+ VRAM    | `exaone3.5:32b` / `qwen3.6:27b`      | 최고 품질 후보. `exaone4.0`은 태그 확인 후 사용 |
| 메모리 부족    | M1/M2 Mac 8GB | `exaone3.5:2.4b` / `gemma4:e2b`      | Metal 가속 활용                       |


#### 🎯 내 최종 추천 (PDF RAG 용도)

**시작은 `exaone3.5:7.8b`로 결정.** 이유는:

1. **한국어 품질**: LG가 한국어/영어 이중언어로 사전학습한 모델이라 한국어 instruction following이 매우 안정적. RAG에서 "주어진 문서만 보고 답하라"는 지시를 잘 따름.
2. **크기 균형**: 7.8B는 8GB VRAM에서 Q4 양자화로 충분히 돌아가고, 응답 속도도 실용적(5~15초/응답).
3. **32K 컨텍스트**: RAG에서 검색된 청크 여러 개 + 대화 히스토리를 담기에 충분.
4. **검증된 실적**: 신·기록 블로그의 실전 테스트에서도 7B로 만족스러운 결과를 보였고, 2.4B도 의외로 쓸만하다고 보고됨.

**대안 시나리오별 추천:**

- 💪 **하드웨어 좋고 최고 품질 원함** → `exaone3.5:32b` 또는 `qwen3.6:27b` (`exaone4.0`은 공식 태그 확인 후 후보로 검토)
  - 긴 문서, 복잡한 추론 필요한 RAG에 강함
- 🐌 **저사양 / 빠른 응답 우선** → `exaone3.5:2.4b`
  - 노트북에서도 돌아감, 단순 사실 조회용으로 충분
- 🌐 **한국어+영어+중국어 등 다국어 PDF** → `qwen3.5:9b`
  - 다국어 균형이 가장 좋음
- 📚 **128K 이상 긴 문서를 통째로 넣고 싶음** → `qwen3.5:32b` 또는 `qwen3.6:27b` (`exaone4.0`은 태그 확인 후 검토)
  - 단, 일반 RAG에서는 청크 검색이 더 효율적
- 🧪 **여러 모델 비교 평가용** → `exaone3.5:7.8b`, `qwen3.5:9b`, `gemma4:e4b` 셋 다 받아 동일 질문으로 테스트

#### 모델 다운로드 명령어

```bash
# 1순위 추천
ollama pull exaone3.5:7.8b

# 비교 평가용 (선택)
ollama pull exaone3.5:2.4b      # 경량 비교
ollama pull qwen3.5:9b           # 다국어 비교
ollama pull gemma4:e4b           # 구글 다국어

# 고성능 환경
ollama pull exaone3.5:32b        # 최고 한국어
# exaone4.0 계열은 Ollama 공식 라이브러리 등재/태그를 먼저 확인한 뒤 사용
# 미등재 시 HuggingFace GGUF → Modelfile로 직접 변환 필요
# ollama pull exaone4.0:32b

# HuggingFace의 한국어 특화 GGUF를 Ollama에 import (Judy 블로그 방식)
# 예: EEVE-Korean, Bllossom-Llama 등
# 1) wget으로 .gguf 다운로드
# 2) Modelfile 작성
# 3) ollama create my-korean-model -f Modelfile
```

#### 모델 선택 시 주의사항

- **라이선스 확인 필수**: Exaone은 EXAONE AI Model License (비상업적 연구 OK, 상업 사용은 별도 협의). 회사에서 쓸 거면 Qwen(Apache 2.0)이나 Gemma(Gemma Terms)가 더 자유로움.
- **양자화 트레이드오프**: Q4_K_M은 품질 손실이 거의 없으면서 메모리는 절반. Q2/Q3는 한국어 품질이 눈에 띄게 떨어지므로 비추.
- **컨텍스트 윈도우 vs 검색 품질**: 컨텍스트가 길다고 RAG가 좋아지는 건 아님. 결국 **검색 단계에서 좋은 청크를 골라내는 게 핵심**이라 7.8B + 좋은 retriever > 32B + 나쁜 retriever인 경우가 많음.

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      [Indexing Phase]                       │
│                                                             │
│  PDF ──► pymupdf4llm (page_chunks=True) ──► MD Documents    │
│                              │                              │
│                              ▼                              │
│            MarkdownHeaderTextSplitter (1차, 구조 보존)      │
│                              │                              │
│                              ▼                              │
│            RecursiveCharacterTextSplitter (2차, 큰 청크만)  │
│                              │                              │
│                              ▼                              │
│                       Chunks (with metadata)                │
│                              │                              │
│              ┌───────────────┴────────────────┐             │
│              ▼                                ▼             │
│      KURE-v1 Embedding              BM25 (kiwipiepy 토큰화) │
│              │                                │             │
│              ▼                                ▼             │
│      ChromaDB (persist)               BM25 인덱스 (메모리)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       [Query Phase]                         │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ├─► KURE-v1 Embedding ─► Vector Search ─┐              │
│      │                                       ├─► Ensemble   │
│      └─► BM25 Search ───────────────────────┘               │
│                              │                              │
│                              ▼                              │
│                    bge-reranker (Top-K 재정렬)              │
│                              │                              │
│                              ▼                              │
│                    Context + 대화기록 + 시스템프롬프트      │
│                              │                              │
│                              ▼                              │
│                       Ollama (exaone3.5)                    │
│                              │                              │
│                              ▼                              │
│                       Streaming Answer                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 단계별 개발 계획

### Phase 1: 환경 구축 (Day 1)

- Ollama 설치 ([https://ollama.com](https://ollama.com))
- 모델 다운로드: `ollama pull exaone3.5:7.8b` (`latest`가 바뀔 수 있으므로 명시 태그 사용)
- Python 가상환경 생성, 의존성 설치
- 테스트용 PDF 준비

### Phase 2: 기본 RAG 파이프라인 (Day 2~3)

- PDF → Markdown 변환 (pymupdf4llm)
- 청킹 + 임베딩 + ChromaDB 저장
- 단일 질의응답 동작 확인

### Phase 3: 대화형 기능 (Day 4)

- Message Manager 클래스 구현 (deque 기반)
- 시스템 프롬프트 설계
- 스트리밍 응답 구현

### Phase 4: 정확도 개선 (Day 5~6)

- 한국어 BM25 토크나이저(`kiwipiepy` 등) 적용
- BM25 + 벡터 하이브리드 검색
- Reranker 추가
- 청크 크기 튜닝 (테스트 케이스 기반)

### Phase 5: 출처 표시 & 안정성 (Day 7)

- 답변 시 페이지 번호 노출
- 검색 결과 없을 때 처리
- 예외 처리 보강

### Phase 6: UI (Day 8~9, 선택)

- Chainlit 우선 적용 (대안: Streamlit)
- PDF 업로드 기능
- 대화 히스토리 표시

### Phase 7: 평가 & 문서화 (Day 10)

- 테스트 케이스 20개 작성
- 정확도/속도 측정
- README 작성

---

## 5. 폴더 구조

```
rag-chatbot/
├── README.md
├── requirements.txt
├── config.py                  # 모델명, 경로 등 설정
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py          # PDF → Markdown
│   ├── indexer.py             # 청킹 + 임베딩 + 저장
│   ├── retriever.py           # 하이브리드 검색 + 리랭킹
│   ├── chat_manager.py        # 대화 히스토리 관리
│   ├── llm_client.py          # Ollama 호출/스트리밍 래퍼 (확장 시 LCEL 전환)
│   └── main.py                # CLI 진입점
├── data/
│   └── pdfs/                  # 원본 PDF 보관
├── chroma_db/                 # 벡터 DB (gitignore)
├── indexes/                   # BM25/문서 manifest 캐시 (gitignore)
├── tests/
│   └── test_cases.py          # 평가용 Q&A 셋
└── app.py                     # Chainlit 또는 Streamlit UI (선택)
```

---

## 6. 핵심 구현 코드

### 6.1 의존성 (`requirements.txt`)

```txt
ollama>=0.3.0
pymupdf4llm>=0.0.17
langchain>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.1.2
langchain-huggingface>=0.1.0
langchain-text-splitters>=0.3.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
kiwipiepy>=0.20.0              # 한국어 BM25 토큰화
FlagEmbedding>=1.2.0          # bge-reranker
chainlit>=1.3.0                # UI (추천)
streamlit>=1.30.0              # UI 대안
pytest>=8.0.0                  # 평가/회귀 테스트

# 평가 단계 선택 의존성
ragas>=0.2.0
langchain-ollama>=0.2.0        # RAGAS/LCEL 전환 시 ChatOllama 사용
```

> `pymupdf4llm`/PyMuPDF 계열은 AGPL 또는 상용 라이선스입니다. 개인 실험·오픈소스가 아니라 회사 내부 서비스나 배포형 제품이면 상용 라이선스 또는 대체 파서를 먼저 검토해야 합니다.

### 6.2 설정 (`config.py`)

```python
from pathlib import Path

# 경로
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
INDEX_DIR = BASE_DIR / "indexes"

# 개발 단계에서는 매 실행마다 벡터 DB를 새로 만들어 중복 인덱싱을 방지.
# 운영 단계에서는 False로 바꾸고 문서 해시 기반 manifest/ids를 사용.
REBUILD_INDEX = True

# 모델
LLM_MODEL = "exaone3.5:7.8b"              # latest 대신 명시 태그 사용
EMBED_MODEL = "nlpai-lab/KURE-v1"         # 한국어 PDF 기본값
ALT_EMBED_MODEL = "BAAI/bge-m3"           # 다국어 PDF 대안
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 실행 장치: cuda / mps / cpu 중 환경에 맞게 선택
EMBED_DEVICE = "cpu"

# 리랭커 FP16 가속: GPU(cuda)에서만 유효. mps/cpu에서는 False 권장.
RERANKER_USE_FP16 = EMBED_DEVICE == "cuda"

# 청킹 (문서 유형에 따라 조정)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# 검색
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# 대화
MAX_HISTORY = 10
```

### 6.3 PDF 로더 (`src/pdf_loader.py`)

```python
import pymupdf4llm
from langchain_core.documents import Document


ALLOWED_METADATA_KEYS = {"source", "page", "title", "h1", "h2", "h3"}


def clean_metadata(metadata: dict) -> dict:
    """Chroma에 안전하게 넣을 수 있는 단순 타입 metadata만 유지."""
    cleaned = {}
    for key, value in metadata.items():
        if key not in ALLOWED_METADATA_KEYS:
            continue
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def load_pdf_with_pages(file_path: str) -> list[Document]:
    """PDF를 페이지 단위 Markdown Document로 변환하고 출처 메타데이터를 유지."""
    page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    docs: list[Document] = []
    for chunk in page_chunks:
        raw_metadata = dict(chunk.get("metadata", {}))
        page = raw_metadata.get("page") or raw_metadata.get("page_number")
        metadata = clean_metadata({
            "source": file_path,
            "page": page,
            "title": raw_metadata.get("title"),
        })
        docs.append(Document(
            page_content=chunk.get("text", ""),
            metadata=metadata,
        ))
    return docs
```

> 페이지별로 `to_markdown()`을 반복 호출하는 방식보다 `page_chunks=True`가 단순하고, 페이지 메타데이터 유지에도 유리합니다. 다만 스캔본 PDF는 OCR 품질을 별도로 확인해야 합니다. Chroma metadata는 단순 타입만 안정적으로 저장되므로 `clean_metadata()`로 필요한 값만 남깁니다.

### 6.4 인덱서 (`src/indexer.py`)

```python
import shutil

from kiwipiepy import Kiwi
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from .pdf_loader import clean_metadata, load_pdf_with_pages
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL,
    MARKDOWN_HEADERS,
    REBUILD_INDEX,
    VECTOR_DB_DIR,
)


kiwi = Kiwi()


def tokenize_korean(text: str) -> list[str]:
    """BM25용 한국어 형태소 토큰화."""
    return [
        token.form
        for token in kiwi.tokenize(text)
        if token.form.strip()
    ]


def split_markdown_documents(docs):
    """Markdown 헤더 구조를 먼저 보존한 뒤 큰 청크만 재분할."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    markdown_docs = []
    for doc in docs:
        split_docs = header_splitter.split_text(doc.page_content)
        for split_doc in split_docs:
            # 헤더 정보(h1/h2/h3)와 원본 메타데이터(source/page)를 모두 보존.
            merged_metadata = {**doc.metadata, **split_doc.metadata}
            split_doc.metadata = clean_metadata(merged_metadata)
            markdown_docs.append(split_doc)

    return recursive_splitter.split_documents(markdown_docs)


def build_index(pdf_paths: list[str]):
    """PDF 리스트를 받아 벡터DB + BM25 인덱스 생성.

    REBUILD_INDEX 동작:
      - True  : 기존 Chroma DB를 삭제하고 chunks로 새로 빌드 (개발 기본값)
      - False : 디스크의 기존 Chroma DB를 재사용 (운영 모드).
                이 모드에서는 chunks를 다시 넣지 않음 — Chroma.from_documents를 호출하면
                기존 컬렉션에 동일 문서가 append 되어 중복 인덱싱이 누적됨.
                ⚠️ 운영 모드의 증분 인덱싱(파일 해시 기반 manifest + 안정적 chunk id)은 추후 구현.
    BM25는 메모리 인덱스라 매 실행마다 chunks에서 재생성.
    """
    all_docs = []
    for path in pdf_paths:
        all_docs.extend(load_pdf_with_pages(path))

    chunks = split_markdown_documents(all_docs)
    print(f"총 청크 수: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    if REBUILD_INDEX:
        # 개발 모드: 기존 DB 삭제 후 chunks로 새로 빌드
        if VECTOR_DB_DIR.exists():
            shutil.rmtree(VECTOR_DB_DIR)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTOR_DB_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        # 운영 모드: 기존 DB 재사용. chunks는 추가하지 않음 (중복 방지).
        # 디스크에 인덱스가 없으면 명시적으로 에러를 내서 의도치 않은 빈 인덱스 사용을 막음.
        if not VECTOR_DB_DIR.exists():
            raise RuntimeError(
                f"REBUILD_INDEX=False인데 {VECTOR_DB_DIR}에 인덱스가 없습니다. "
                f"최초 1회는 REBUILD_INDEX=True로 실행하거나, 증분 인덱싱 로직을 구현하세요."
            )
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    bm25 = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize_korean,
        k=10,
    )

    return vectorstore, bm25, chunks
```

> MVP에서는 `REBUILD_INDEX=True`로 매 실행마다 벡터 DB를 새로 만들어 중복 인덱싱을 피합니다. 운영 단계로 가면 `REBUILD_INDEX=False`로 전환해 기존 인덱스를 재사용하고(`Chroma.from_documents`가 아닌 `Chroma()` 생성자 사용), 별도로 PDF 파일 해시/수정시간을 저장하는 manifest와 안정적인 chunk id 기반 증분 인덱싱을 추가 구현해야 합니다 — 그래야 변경된 문서만 갱신할 수 있습니다. Chroma는 디스크에 남지만 BM25는 메모리 인덱스라 재시작 시 청크에서 재생성됩니다(필요 시 `indexes/`에 직렬화 캐시).

### 6.5 하이브리드 검색 + 리랭킹 (`src/retriever.py`)

```python
from langchain.retrievers import EnsembleRetriever
from FlagEmbedding import FlagReranker
from config import (
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    RERANKER_MODEL,
    RERANKER_USE_FP16,
)

class HybridRetriever:
    def __init__(self, vectorstore, bm25):
        bm25.k = TOP_K_RETRIEVAL
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )
        self.ensemble = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )
        # FP16은 GPU(cuda)에서만 유효. config의 EMBED_DEVICE와 자동 연동됨.
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=RERANKER_USE_FP16)
    
    def retrieve(self, query: str):
        """1차 검색(앙상블) → 2차 리랭킹"""
        candidates = self.ensemble.invoke(query)
        
        if not candidates:
            return []
        
        # 리랭커로 재정렬
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]
        
        # score 기준 정렬
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [doc for doc, _ in ranked[:TOP_K_RERANK]]
```

### 6.6 대화 관리 (`src/chat_manager.py`)

```python
"""신·기록 블로그의 Message Manager 패턴 차용"""
from collections import deque
from config import MAX_HISTORY

SYSTEM_PROMPT = """당신은 문서 기반 한국어 질의응답 어시스턴트입니다.

규칙:
1. 제공된 [문서 내용]만을 근거로 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
3. 답변은 명확하고 간결하게, 한국어로 작성하세요.
4. 가능하면 답변 끝에 참고한 페이지를 (p.X) 형식으로 표기하세요.
5. 표/숫자가 포함된 경우 원문을 그대로 인용하세요."""

class ChatManager:
    def __init__(self):
        # MAX_HISTORY는 "대화 턴" 단위. 한 턴 = user 메시지 + assistant 메시지 2개.
        # 따라서 deque maxlen은 MAX_HISTORY * 2.
        self.history = deque(maxlen=MAX_HISTORY * 2)
        self.system_prompt = SYSTEM_PROMPT
    
    def add_user(self, content: str):
        self.history.append({"role": "user", "content": content})
    
    def add_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})
    
    def build_messages(self, context: str, query: str):
        """LLM에 보낼 메시지 배열 구성.

        ⚠️ 호출 계약 (중요):
        이 메서드는 현재 query를 반환 메시지에 직접 끼워 넣지만 history에는 추가하지 않습니다.
        호출자가 LLM 응답을 받은 후 add_user(query) → add_assistant(answer) 순으로
        history에 직접 기록해야 합니다. 이 순서가 바뀌면 query가 중복 전송됩니다.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"[문서 내용]\n{context}"},
        ]
        # 이전 대화 히스토리 (현재 query는 아직 history에 없음)
        messages.extend(list(self.history))
        # 현재 질문을 메시지 끝에 추가 (history에는 호출자가 응답 후에 기록)
        messages.append({"role": "user", "content": query})
        return messages
```

### 6.7 LLM 클라이언트 (`src/llm_client.py`)

```python
"""Ollama 호출을 캡슐화. 추후 LCEL/ChatOllama 전환 시 이 파일만 교체하면 됨.

MVP에서는 ollama-python을 직접 사용해 단순함을 유지하고,
확장 단계(평가/RAGAS/멀티프로바이더 등)에서 langchain_ollama.ChatOllama로 교체.
"""
from typing import Iterable

import ollama

from config import LLM_MODEL


def stream_chat(messages: list[dict]) -> Iterable[str]:
    """messages 리스트를 Ollama에 전달하고 응답 텍스트를 청크 단위로 yield."""
    for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
        text = chunk["message"]["content"]
        if text:
            yield text


def chat(messages: list[dict]) -> str:
    """비스트리밍 호출 (평가나 배치 처리용). 전체 응답을 문자열로 반환."""
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]
```

> LLM 호출을 한 곳으로 모아두면 (1) 모델 교체, (2) 토큰 사용량 로깅, (3) 재시도 정책 추가, (4) LCEL 체인으로 전환할 때 변경 범위가 이 파일에 국한됩니다.

### 6.8 메인 실행 (`src/main.py`)

```python
import time
from pathlib import Path

from .indexer import build_index
from .retriever import HybridRetriever
from .chat_manager import ChatManager
from .llm_client import stream_chat
from config import DATA_DIR

def format_context(docs) -> str:
    """검색된 문서를 LLM 컨텍스트용으로 포맷"""
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[문서 {i} | p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)

def main():
    # 1. 인덱스 구축
    print("📚 PDF 인덱싱 중...")
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ {DATA_DIR}에 PDF 파일이 없습니다.")
        return
    
    vectorstore, bm25, _ = build_index([str(p) for p in pdf_files])
    retriever = HybridRetriever(vectorstore, bm25)
    chat = ChatManager()
    
    print("✅ 준비 완료! 질문을 입력하세요 (종료: 'exit')\n")
    
    # 2. 대화 루프
    while True:
        query = input("👤 > ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "종료"):
            break
        
        start = time.time()
        
        # 검색
        docs = retriever.retrieve(query)
        if not docs:
            print("🤖 관련 문서를 찾을 수 없습니다.\n")
            continue
        
        context = format_context(docs)
        messages = chat.build_messages(context, query)
        
        # 스트리밍 답변 (llm_client 래퍼 통해 호출)
        print("🤖 ", end="", flush=True)
        answer = ""
        for text in stream_chat(messages):
            print(text, end="", flush=True)
            answer += text
        
        # 히스토리 기록 (build_messages 호출 계약 준수)
        chat.add_user(query)
        chat.add_assistant(answer)
        
        elapsed = time.time() - start
        print(f"\n⏱️  {elapsed:.1f}초\n")

if __name__ == "__main__":
    main()
```

---

## 7. 정확도 개선 전략

답변 품질이 낮을 때 점검 순서:

### 7.1 검색 단계 (가장 중요)

1. **청크 크기 튜닝**: 동일 질문으로 chunk_size를 400/600/800/1200으로 바꿔가며 비교
2. **Top-K 조정**: 너무 적으면 누락, 너무 많으면 노이즈. 보통 리랭킹 후 3~5개가 적절
3. **메타데이터 필터**: 페이지/섹션 정보로 필터링 가능하게

### 7.2 임베딩 단계

- 한국어 PDF는 `KURE-v1`을 기본값으로 사용
- 한·영 혼용이나 다국어 문서가 많으면 `bge-m3`와 동일 테스트셋으로 비교
- 도메인 특화 데이터가 많으면 임베딩 모델 fine-tuning 고려

### 7.3 프롬프트 단계

- "근거가 없으면 모른다고 답하라"를 강하게 명시 (환각 방지)
- Few-shot 예시 추가 (특히 출처 표기 형식)

### 7.4 평가 방법

간단한 평가 셋을 만들어 변경 전후 비교:

```python
# tests/test_cases.py 예시
TEST_CASES = [
    {"q": "연장근로수당은 통상임금의 몇 %인가?", "expected_keywords": ["50%", "100분의 50"]},
    {"q": "근로시간은 1주 몇 시간을 초과할 수 없는가?", "expected_keywords": ["40시간"]},
    # ...
]
```

> **RAGAS 자동 평가 (선택)**: 키워드 매칭만으로는 답변 품질을 충분히 측정하기 어려우므로, 평가용 LLM 기반 자동 점수가 필요하면 RAGAS를 병행합니다. 외부 API 독립이 목표이므로 평가용 LLM은 `langchain_ollama.ChatOllama(model="exaone3.5:7.8b")`로 고정해 모든 평가를 로컬에서 수행합니다. RAGAS는 Faithfulness(답변이 컨텍스트에 충실한가), Answer Relevancy(질문과의 관련성), Context Precision(검색된 컨텍스트의 정확도) 등을 자동 산출합니다. 단, 평가 LLM 자체가 한국어에 약하면 점수 신뢰도가 낮아지므로 동일 테스트셋으로 사람 평가와 한 번 캘리브레이션이 필요합니다.

---

## 8. 트러블슈팅 체크리스트


| 증상         | 원인 후보              | 해결                                             |
| ---------- | ------------------ | ---------------------------------------------- |
| 한글이 깨져 추출됨 | PDF가 이미지 기반(스캔본)   | OCR 필요 (`tesseract`, `unstructured`)           |
| 표 내용이 엉망   | 청크 크기가 작아 표가 잘림    | chunk_size 키우거나 표 전용 파서 사용                     |
| 답변이 엉뚱함    | 검색 단계에서 잘못된 청크 가져옴 | 리랭커 추가, BM25 가중치 조정                            |
| 너무 느림      | 7B 모델 + CPU        | 양자화 모델 사용 (Q4_K_M), GPU 활용                     |
| GPU 메모리 부족 | 임베딩 + LLM 동시 로드    | 임베딩은 CPU로 내리고 LLM만 GPU 사용, 필요 시 더 작은 임베딩 모델 비교 |
| 대화 흐름이 끊김  | 히스토리가 컨텍스트 윈도우 초과  | MAX_HISTORY 줄이기                                |


---

## 9. 실행 절차

```bash
# 1. Ollama 실행 확인
ollama serve  # 별도 터미널 또는 백그라운드

# 2. 모델 다운로드
ollama pull exaone3.5:7.8b

# 3. 의존성 설치
pip install -r requirements.txt

# 4. PDF 배치
cp your_document.pdf data/pdfs/

# 5. 실행
python -m src.main
```

---

## 10. 확장 아이디어 (이후 단계)

- **멀티 PDF 컬렉션**: 문서별 컬렉션 분리, 어떤 문서에서 답이 나왔는지 표시
- **음성 입력/출력**: Whisper + TTS 결합
- **테이블 QA 강화**: `unstructured`로 표를 별도 추출 후 별도 인덱스
- **그래프 RAG**: LightRAG, GraphRAG 도입 (관계 추론 강화)
- **Agent화**: 검색 결과가 부족하면 추가 검색하도록 ReAct 패턴 적용
- **API 서버화**: FastAPI로 감싸서 외부 서비스에서 호출 가능하게

---

## 참고 자료

- 신·기록 블로그: [https://god-logger.tistory.com/205](https://god-logger.tistory.com/205) — pymupdf4llm 마크다운 변환, Message Manager 패턴
- Judy 블로그: [https://velog.io/@judy_choi/LLaMA3-을-이용한-RAG-구축-Ollama-사용법-정리](https://velog.io/@judy_choi/LLaMA3-을-이용한-RAG-구축-Ollama-사용법-정리) — LangChain LCEL 체인, gguf 모델 Ollama 적용
- Ollama 공식: [https://ollama.com/](https://ollama.com/)
- KURE-v1: [https://huggingface.co/nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1)
- BGE-M3: [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- LangChain Chroma: [https://docs.langchain.com/oss/python/integrations/vectorstores/chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- LangChain BM25: [https://docs.langchain.com/oss/python/integrations/retrievers/bm25/](https://docs.langchain.com/oss/python/integrations/retrievers/bm25/)
- LangChain RAG 튜토리얼: [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)
