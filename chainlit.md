# Ollama PDF RAG Chatbot

로컬 PDF 문서를 기준으로 답변하는 한국어 RAG 챗봇입니다.

질문을 입력하면 `data/pdfs/`의 문서를 검색하고, 답변 마지막에 실제 metadata 기반 출처를 표시합니다.

PDF를 추가하려면 채팅창에 `/upload`를 입력하세요. 업로드 후 전체 PDF 기준으로 인덱스를 다시 생성합니다.
