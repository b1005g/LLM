import pandas as pd
import openai
from openai import OpenAI
import os

# 1. API 키 읽기
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# 2. API 키 설정

api_key = load_api_key("api_key.txt")
openai.api_key = api_key
client = OpenAI(api_key=api_key)

path = os.getcwd()
train = pd.read_csv("APEACH_데이터_라벨링완료.csv", encoding ='utf-8-sig')

sampled_data = train.groupby("label_numbers").first().reset_index()
sampled_data = sampled_data.drop(columns = ['is_immoral','label_words'])

samples = [{"text": text, "label": label[1]} for text, label in zip(sampled_data['text'], sampled_data['label_numbers'])]
text = train['text']
label = train['label_numbers']

#prompt 양식
categories = [
    "모욕 (0)", "협박 (1)", "욕설 (2)", "인종차별 (3)", 
    "성차별 (4)", "성희롱 (5)", "장애인 차별 (6)", 
    "혐오발언 (7)", "종교차별 (8)", "혐오발언 아님 (9)"
]
def generate_prompt(text):
    return (
        f"다음 문장을 읽고, 해당 문장이 아래 카테고리 중 어떤 것에 해당하는지 판별하세요:\n"
        f"문장: {text}\n\n"
        f"카테고리: {', '.join(categories)}\n\n"
        f"답변 형식: 카테고리 번호와 이름으로만 답변하세요. 예: '0 모욕'"
    )

def classify_text(text):
    prompt = generate_prompt(text)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

label_result = []
for text in train['text']:
    result = classify_text(text)
    label_result.append((text, result))

predict = pd.DataFrame(label_result, columns=['text', 'label'])
predict['label'] = predict['label'].apply(lambda x: int(x.strip().split()[0].replace("'", "")))
print(predict)