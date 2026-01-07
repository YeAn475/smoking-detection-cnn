import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# 페이지 설정
st.set_page_config(
    page_title="흡연 감지 AI",
    page_icon="🚭",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .smoking-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .no-smoking {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# 모델 로드 (캐싱)
@st.cache_resource
def load_smoking_model():
    """저장된 모델을 로드합니다."""
    model_paths = [
        './model/smoking_classification_model.h5',
        './model/smoking_classification_model.keras',
        './model/smoking_detection_model.h5',
        './model/best_smoking_model.keras',
        './model/best_smoking_model.h5',
        'smoking_classification_model.h5',
        'smoking_detection_model.h5',
        'best_smoking_model.keras',
        'model.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                st.sidebar.success(f"✅ 모델 로드 성공: {path}")
                return model
            except Exception as e:
                st.sidebar.warning(f"로드 시도 실패 ({path}): {e}")
                continue
    
    st.error("모델 파일을 찾을 수 없습니다.")
    return None

# 클래스 정보
class_names = ['notsmoking', 'smoking']
class_names_kr = {'notsmoking': '비흡연', 'smoking': '흡연'}
class_icons = {'notsmoking': '✅', 'smoking': '🚬'}

# 이미지 크기
IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image(image):
    """이미지 전처리 함수"""
    img = image.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)
    return img_array

def preprocess_frame(frame):
    """동영상 프레임 전처리 함수"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_array = np.expand_dims(img, 0)
    return img_array

def predict_smoking(model, img_array):
    """흡연 여부 예측"""
    prediction = model.predict(img_array, verbose=0)
    return prediction

def get_prediction_result(prediction):
    """예측 결과 해석"""
    pred_array = np.array(prediction).flatten()
    
    if len(pred_array) == 1:
        smoking_prob = float(pred_array[0])
        notsmoking_prob = 1.0 - smoking_prob
        probs = [notsmoking_prob, smoking_prob]
    elif len(pred_array) == 2:
        probs = [float(pred_array[0]), float(pred_array[1])]
    else:
        probs = [0.5, 0.5]
    
    max_index = 0 if probs[0] > probs[1] else 1
    predicted_class = class_names[max_index]
    confidence = probs[max_index] * 100
    
    return predicted_class, confidence, probs

# 메인 타이틀
st.markdown('<h1 class="main-title">🚭 흡연 감지 AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">이미지 또는 동영상을 업로드하면 AI가 흡연 여부를 감지합니다!</p>', unsafe_allow_html=True)

st.markdown("---")

# 설명
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info("📌 **흡연 감지 AI 시스템**\n\n어린이보호구역, 금연구역 등에서 흡연 행위를 자동으로 감지하여 과태료 및 경고를 부과하기 위한 AI 시스템입니다.")

with col_info2:
    st.warning("⚠️ **분류 가능한 클래스**\n\n- **smoking (흡연)**: 흡연 중인 이미지\n- **notsmoking (비흡연)**: 흡연하지 않는 이미지")

st.markdown("---")

# ===================== 이미지 업로드 섹션 =====================
st.markdown("### 📷 이미지를 업로드하세요")
uploaded_file = st.file_uploader(
    "이미지 파일 선택",
    type=["jpg", "jpeg", "png"],
    help="JPG, JPEG, PNG 형식의 이미지를 업로드하세요.",
    key="image_uploader"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📷 업로드한 이미지")
        st.image(image, use_container_width=True, caption="업로드된 이미지")
    
    with col2:
        st.markdown("#### 🔍 예측 결과")
        
        model = load_smoking_model()
        
        if model is not None:
            img_array = preprocess_image(image)
            
            with st.spinner("🔄 AI가 분석 중입니다..."):
                prediction = predict_smoking(model, img_array)
            
            predicted_class, confidence, probs = get_prediction_result(prediction)
            
            st.markdown("##### 클래스별 확률:")
            for i, class_name in enumerate(class_names):
                prob = probs[i] * 100
                if class_name == predicted_class:
                    st.markdown(f"**{class_icons[class_name]} {class_name}({class_names_kr[class_name]}): {prob:.2f}%**")
                else:
                    st.markdown(f"{class_icons[class_name]} {class_name}({class_names_kr[class_name]}): {prob:.2f}%")
                st.progress(float(prob / 100))
            
            st.markdown("---")
            
            if predicted_class == 'smoking':
                st.markdown(f"""
                <div class="result-box smoking-detected">
                    <h3>🚬 흡연 감지!</h3>
                    <p>신뢰도: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ **경고**: 흡연 감지! 과태료 부과 대상입니다.")
            else:
                st.markdown(f"""
                <div class="result-box no-smoking">
                    <h3>✅ 비흡연 확인</h3>
                    <p>신뢰도: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ 흡연이 감지되지 않았습니다.")

st.markdown("---")

# ===================== 동영상 업로드 섹션 =====================
st.markdown("### 🎬 동영상을 업로드하세요")
uploaded_video = st.file_uploader(
    "동영상 파일 선택",
    type=["mp4", "avi", "mov", "mkv"],
    help="MP4, AVI, MOV, MKV 형식의 동영상을 업로드하세요.",
    key="video_uploader"
)

if uploaded_video is not None:
    # 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    st.markdown("#### 🎬 업로드한 동영상")
    st.video(uploaded_video)
    
    st.markdown("---")
    
    # 분석 옵션
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        frame_skip = st.slider("프레임 간격 (높을수록 빠름)", 1, 30, 10, 
                               help="몇 프레임마다 분석할지 설정합니다.")
    with col_opt2:
        max_frames = st.slider("최대 분석 프레임 수", 10, 200, 50,
                               help="분석할 최대 프레임 수를 설정합니다.")
    
    if st.button("🔍 동영상 분석 시작", type="primary"):
        model = load_smoking_model()
        
        if model is not None:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.info(f"📊 동영상 정보: {total_frames} 프레임, {fps} FPS, 약 {total_frames/fps:.1f}초")
            
            # 분석 결과 저장
            results = []
            smoking_frames = []
            frame_count = 0
            analyzed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 결과 표시 영역
            result_container = st.container()
            
            while cap.isOpened() and analyzed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # 프레임 분석
                    img_array = preprocess_frame(frame)
                    prediction = predict_smoking(model, img_array)
                    predicted_class, confidence, probs = get_prediction_result(prediction)
                    
                    results.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'class': predicted_class,
                        'confidence': confidence
                    })
                    
                    # 흡연 감지된 프레임 저장
                    if predicted_class == 'smoking' and confidence > 60:
                        smoking_frames.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'confidence': confidence,
                            'image': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        })
                    
                    analyzed_count += 1
                    progress = analyzed_count / max_frames
                    progress_bar.progress(progress)
                    status_text.text(f"분석 중... {analyzed_count}/{max_frames} 프레임 ({frame_count}/{total_frames})")
                
                frame_count += 1
            
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("✅ 분석 완료!")
            
            # 결과 요약
            st.markdown("---")
            st.markdown("### 📊 분석 결과")
            
            smoking_count = sum(1 for r in results if r['class'] == 'smoking')
            notsmoking_count = len(results) - smoking_count
            smoking_ratio = (smoking_count / len(results)) * 100 if results else 0
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("총 분석 프레임", f"{len(results)}개")
            with col_r2:
                st.metric("흡연 감지 프레임", f"{smoking_count}개", 
                         delta=f"{smoking_ratio:.1f}%")
            with col_r3:
                st.metric("비흡연 프레임", f"{notsmoking_count}개")
            
            # 최종 판정
            st.markdown("---")
            if smoking_ratio > 30:
                st.markdown("""
                <div class="result-box smoking-detected">
                    <h3>🚬 흡연 행위 감지!</h3>
                    <p>동영상에서 흡연 행위가 감지되었습니다.</p>
                </div>
                """, unsafe_allow_html=True)
                st.error(f"⚠️ 흡연 감지 비율: {smoking_ratio:.1f}% - 과태료 부과 대상입니다.")
            elif smoking_ratio > 10:
                st.warning(f"⚠️ 흡연 의심: {smoking_ratio:.1f}% - 추가 확인이 필요합니다.")
            else:
                st.markdown("""
                <div class="result-box no-smoking">
                    <h3>✅ 흡연 미감지</h3>
                    <p>동영상에서 흡연 행위가 감지되지 않았습니다.</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ 흡연이 감지되지 않았습니다.")
            
            # 흡연 감지된 프레임 표시
            if smoking_frames:
                st.markdown("---")
                st.markdown("### 🚬 흡연 감지 프레임")
                
                cols = st.columns(min(4, len(smoking_frames)))
                for idx, sf in enumerate(smoking_frames[:8]):  # 최대 8개만 표시
                    with cols[idx % 4]:
                        st.image(sf['image'], caption=f"⏱️ {sf['time']:.1f}초 ({sf['confidence']:.1f}%)")
            
            # 타임라인 그래프
            if results:
                st.markdown("---")
                st.markdown("### 📈 시간별 분석 결과")
                
                import pandas as pd
                df = pd.DataFrame(results)
                df['smoking_score'] = df.apply(
                    lambda x: x['confidence'] if x['class'] == 'smoking' else 100 - x['confidence'], 
                    axis=1
                )
                
                st.line_chart(df.set_index('time')['smoking_score'])
                st.caption("📌 값이 높을수록 흡연 확률이 높음 (50 이상 = 흡연 감지)")
    
    # 임시 파일 삭제
    try :
        os.unlink(video_path)
    except :
        pass
# 사이드바
st.sidebar.markdown("## 📌 프로젝트 정보")
st.sidebar.markdown("""
**AI기반 흡연자 감지 시스템**

- **목적**: 어린이보호구역 및 금연구역 흡연 감지
- **모델**: CNN (Convolutional Neural Network)
- **분류 클래스**: smoking, notsmoking
- **입력 이미지**: 224x224 RGB
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 사용 방법")
st.sidebar.markdown("""
**이미지 분석**
1. 이미지 업로드
2. AI 분석 결과 확인

**동영상 분석**
1. 동영상 업로드
2. 분석 옵션 설정
3. '분석 시작' 버튼 클릭
4. 프레임별 결과 확인
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 프로젝트")
st.sidebar.markdown("인공지능개발 양성과정 - 딥러닝 프로젝트")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🚭 AI기반 흡연자 감지 예측모델 개발 및 시각화</p>
    <p>인공지능개발 양성과정 딥러닝 산출물</p>
</div>
""", unsafe_allow_html=True)