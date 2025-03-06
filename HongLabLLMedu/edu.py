
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("shubhammaindola/harry-potter-books")
# print("Path to dataset files:", path)
#
# # Download latest version
# path = kagglehub.dataset_download("leelatte/alicetxt")
# print("Path to dataset files:", path)

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import re, os

# 1. BPE 모델을 사용해 토크나이저 인스턴스를 생성합니다.
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# 공백 기준으로 단어를 분리하는 전처리 과정을 설정합니다.
tokenizer.pre_tokenizer = Whitespace()
# 3. BPE Trainer를 설정하여 특수 토큰들을 지정합니다.
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# 4. 준비된 코퍼스를 사용해 토크나이저를 학습시킵니다.

def clean_text(path, filename):
    with open(path + filename, 'r', encoding='utf-8') as file:
        book_text = file.read()

    cleaned_text = re.sub(r'\n+', ' ', book_text) # 줄바꿈을 빈칸으로 변경
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # 여러 빈칸을 하나의 빈칸으로

    print(path + "cleaned_" + filename, len(cleaned_text), "characters") # 글자 수 출력

    tokenizer.train_from_iterator(cleaned_text, trainer=trainer)
    with open(path + "cleaned_" + filename, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    return path + "cleaned_" + filename


folder_path = "./shubhammaindola/harry-potter-books/versions/1/"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
cleaned_file_list = [clean_text(folder_path, file) for file in file_list]


# 5. 학습된 토크나이저를 사용하여 입력 텍스트를 토큰화합니다.
encoded = tokenizer.encode("Hello, I am testing the Tokenizers library!")
print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
