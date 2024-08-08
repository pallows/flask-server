import json
import logging
from flask import Flask, request, current_app
from collections import OrderedDict
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def api_response(status, message, data):
    response_data = OrderedDict([
        ('status', status),
        ('message', message),
        ('data', data)
    ])
    response_json = json.dumps(response_data, indent=4)
    logging.info("Sending response: %s", response_json)
    return current_app.response_class(response_json, mimetype='application/json')

def convert_to_dataframe(data):
    return pd.DataFrame(data)

def delete_duplicate(row):
    return list(set(row)) if isinstance(row, list) else row

def delete_space(row):
    return [item.replace(" ", "") for item in row] if isinstance(row, list) else row

def preprocess_and_classify(df, text_columns):
    for col in text_columns:
        if col in ['interest', 'ideal']:
            df[col] = df[col].apply(lambda x: ' '.join(delete_space(delete_duplicate(x.split(',')))))  # 리스트를 문자열로 변환
        else:
            df[col] = df[col].apply(delete_duplicate)
            df[col] = df[col].apply(delete_space)
            df[col] = df[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    return df

def encode_features(df, text_columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # interest와 ideal 컬럼만 추출
    interest_ideal_data = df[text_columns]
    
    # 원-핫 인코딩
    encoded_features = encoder.fit_transform(interest_ideal_data)
    
    return encoded_features, encoder

def apply_svd(encoded_features, n_components):
    svd = TruncatedSVD(n_components=n_components)
    latent_matrix = svd.fit_transform(encoded_features)
    logging.info("Latent matrix shape: %s", latent_matrix.shape)
    return latent_matrix

def calculate_similarity(latent_matrix, profile_idx):
    sim = cosine_similarity(latent_matrix, latent_matrix)
    return sim[profile_idx, :].reshape(-1)

def get_similar_mbti(mbti):
    mbti_mapping = {
        'ISTJ': ['ENFP', 'ENTP', 'ISFP', 'INFP'],
        'ISFJ': ['ENTP', 'ENFP', 'INTP', 'ISTP'],
        'INFJ': ['ESTP', 'ESFP', 'ISTP', 'INTP'],
        'INTJ': ['ESTJ', 'ENTJ', 'INTJ', 'ISTJ'],
        'ISTP': ['ENFJ', 'ESFJ', 'INFJ', 'ISFJ'],
        'ISFP': ['ENTJ', 'ESTJ', 'INTJ', 'ISTJ'],
        'INFP': ['ENTP', 'ESFJ', 'INTJ', 'ENFP'],
        'INTP': ['ESFJ', 'ENFJ', 'ISFJ', 'INFJ'],
        'ESTP': ['INFJ', 'INTJ', 'ENFJ', 'ENTJ'],
        'ESFP': ['INTJ', 'INFJ', 'ENTJ', 'ENFJ'],
        'ENFP': ['ISTJ', 'ISFJ', 'ESFJ', 'ESTJ'],
        'ENTP': ['ISFJ', 'INTJ', 'ENTP', 'ESTJ'],
        'ESTJ': ['INFP', 'ISFP', 'INTP', 'ENTP'],
        'ESFJ': ['INTP', 'ISTP', 'ENTP', 'ENFP'],
        'ENFJ': ['ISTP', 'INTP', 'ESTP', 'ESFP'],
        'ENTJ': ['ISFP', 'INFP', 'ESFP', 'ESTP']
    }
    return mbti_mapping.get(mbti, [])

def get_results(data, id, text_columns):
    df = convert_to_dataframe(data)
    
    profile = df[df['id'] == id]
    if profile.empty:
        return []

    user_relationship = profile.iloc[0]['relationship']
    user_mbti = profile.iloc[0]['mbti']
    similar_mbtis = get_similar_mbti(user_mbti)
    similar_mbtis.append(user_mbti)  # 자기 자신의 mbti도 포함 유사도 계산시를 위함.
    
    if user_relationship is None:
        df = df[(df['relationship'].isna() | df['relationship'].isnull()) & (df['mbti'].isin(similar_mbtis))]
    else:
        df = df[(df['relationship'] == user_relationship) & (df['mbti'].isin(similar_mbtis))]

    df.reset_index(drop=True, inplace=True)

    # 전처리 및 분류
    df = preprocess_and_classify(df, text_columns)

    profile_idx = df[df['id'] == id].index.values
    if len(profile_idx) == 0 or profile_idx[0] >= len(df):
        raise ValueError(f"Profile index {profile_idx} is out of bounds for the dataframe size {len(df)}")

    user_count = len(df)
    if user_count <= 100:
        n_components = 10
    elif user_count <= 1000:
        n_components = 25
    else:
        n_components = 50
        
    logging.info(user_count)

    # 원-핫 인코딩
    encoded_features, encoder = encode_features(df, text_columns)
    latent_matrix = apply_svd(encoded_features, n_components)
    similarities = calculate_similarity(latent_matrix, profile_idx[0])

    df['similarity'] = similarities
    df = df.sort_values(by="similarity", ascending=False)
    top_10 = df.head(10)

    top_10_ids = top_10['id'].tolist()
    top_10_similarities = top_10['similarity'].tolist()

    # 유사도를 로그로 출력
    for idx, sim in zip(top_10_ids, top_10_similarities):
        logging.info(f"ID: {idx}, Similarity: {sim}")

    return top_10_ids


@app.route('/api/profile/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        logging.info("Received data: %s", json.dumps(data, indent=4))
        
        my_profile_id = data.get('id')
        profile_list = data.get('profiles')
        
        if my_profile_id is None or profile_list is None:
            logging.error("Missing 'id' or 'profiles' in the request data")
            return api_response(400, "Missing 'id' or 'profiles'", {}), 400

        text_columns = ['alcohol', 'education', 'ideal', 'interest', 'jobs', 'personality', 'pros', 'religion', 'smoking']
        
        df_result = get_results(profile_list, my_profile_id, text_columns)
        
        logging.info("Recommendation result: %s", json.dumps({"sortedIdList": df_result}, indent=4))
        
        return api_response(200, "추천 목록 조회 성공", {"sortedIdList": df_result}), 200
    except KeyError as e:
        logging.error("KeyError: %s", str(e))
        return api_response(400, f"Missing key: {str(e)}", {}), 400
    except Exception as e:
        logging.error("Exception: %s", str(e))
        return api_response(500, f"Server error: {str(e)}", {}), 500

@app.route('/')
def user():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, debug=True)
