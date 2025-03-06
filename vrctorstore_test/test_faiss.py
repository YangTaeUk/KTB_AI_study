import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === 1. API 명세 데이터 준비 ===
documents = [
    "User API: Create, Read, Update, Delete",
    "Product API: Add, Remove, Update",
    "Order API: Create, Track, Cancel",
    "Authentication API: Register, Login, Logout",
    "Payment API: Process, Refund, Cancel",
    "Inventory API: Add, Remove, Check Stock",
]

# === 2. 문장 벡터 변환 (임베딩 생성) ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # 가벼운 Transformer 모델
embeddings = model.encode(documents)  # 모든 API 문서 벡터화

# === 3. FAISS 인덱스 생성 및 데이터 추가 ===
d = embeddings.shape[1]  # 벡터 차원
index = faiss.IndexFlatL2(d)  # L2 거리 기반 인덱스 생성
index.add(np.array(embeddings).astype('float32'))  # 벡터 삽입

print(f"✅ FAISS Index 생성 완료! 저장된 API 개수: {index.ntotal}\n")


# === 4. 사용자 입력 쿼리 및 검색 ===
def search_similar_api(user_input, top_k=1):
    """사용자의 자연어 입력을 벡터화하여 유사한 API 명세 추천"""
    query_embedding = model.encode([user_input])  # 입력 문장 벡터화
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)  # FAISS 검색

    # 결과 출력
    print(f"🔍 사용자 입력: {user_input}")
    for i in range(top_k):
        print(f"⭐ 추천 API: {documents[indices[0][i]]} (유사도 거리: {distances[0][i]:.4f})\n")


# === 5. 테스트: 예제 입력 실행 ===
search_similar_api("회원가입 API 필요")  # 예상 결과: Authentication API
search_similar_api("결제 취소 API")  # 예상 결과: Payment API
search_similar_api("상품 추가 기능")  # 예상 결과: Product API
