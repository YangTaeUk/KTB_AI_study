from sentence_transformers import SentenceTransformer
import faiss

# 데이터 준비
texts = [
    "안녕하세요, 반갑습니다.",
    "AI는 정말 재밌어요.",
    "벡터 검색을 테스트 중입니다.",
    "하이, 잘 지내세요?"
]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_numpy=True)
d = embeddings.shape[1]

# 인덱스 생성 및 저장
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, d)
index.add(embeddings)
faiss.write_index(faiss.index_gpu_to_cpu(index), "./vector_index.faiss")

# 인덱스 불러오기
loaded_index = faiss.read_index("vector_index.faiss")
gpu_index = faiss.index_cpu_to_gpu(res, 0, loaded_index)

# 검색
query_text = "AI가 재밌네요."
query_embedding = model.encode([query_text], convert_to_numpy=True)
k = 2
distances, indices = gpu_index.search(query_embedding, k)

# 결과 출력
for i, idx in enumerate(indices[0]):
    print(f"결과 {i+1}: {texts[idx]} (거리: {distances[0][i]})")