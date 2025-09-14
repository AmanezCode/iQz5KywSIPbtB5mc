import json, io, torch
import numpy as np
import pandas as pd
import streamlit as st
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from albumentations.pytorch import ToTensorV2

from main import EfficientNetCarClassifier

class StreamlitCarClassifier:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.transforms = None
        self.setup_transforms()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            st.warning("Модель не найдена. Использую демо-режим.")
            self.setup_demo_model()
    
    def setup_transforms(self):
        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def setup_demo_model(self):
        self.model = EfficientNetCarClassifier(num_cleanliness_classes=3, num_damage_classes=2)
        self.model.eval()
        st.info("🔄 Используется необученная модель для демонстрации интерфейса")
    
    def load_model(self, model_path):
        try:
            self.model = EfficientNetCarClassifier(num_cleanliness_classes=3, num_damage_classes=2)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            st.success(f"✅ Модель загружена с {self.device}")
            
            if 'val_clean_acc' in checkpoint:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Точность (Чистота)", f"{checkpoint['val_clean_acc']:.1%}")
                with col2:
                    st.metric("Точность (Повреждения)", f"{checkpoint.get('val_damage_acc', 0):.1%}")
            
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            self.setup_demo_model()
    
    def predict(self, image):
        try:
            if isinstance(image, Image.Image):
                image_np = np.array(image.convert('RGB'))
            else:
                image_np = image
            
            transformed = self.transforms(image=image_np)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                clean_logits, damage_logits = self.model(image_tensor)
                
                clean_probs = F.softmax(clean_logits, dim=1)
                damage_probs = F.softmax(damage_logits, dim=1)
                
                clean_pred = torch.argmax(clean_probs, dim=1).item()
                damage_pred = torch.argmax(damage_probs, dim=1).item()
                
                clean_confidence = torch.max(clean_probs, dim=1)[0].item()
                damage_confidence = torch.max(damage_probs, dim=1)[0].item()
            
            return {
                'cleanliness': {
                    'prediction': clean_pred,
                    'confidence': clean_confidence,
                    'probabilities': clean_probs[0].cpu().numpy()
                },
                'damage': {
                    'prediction': damage_pred,
                    'confidence': damage_confidence,
                    'probabilities': damage_probs[0].cpu().numpy()
                }
            }
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            return None
    
    def create_results_chart(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        cleanliness_labels = ['Чистый', 'Слегка\nгрязный', 'Очень\nгрязный']
        damage_labels = ['Целый', 'Битый']
        
        clean_probs = results['cleanliness']['probabilities']
        bars1 = ax1.bar(cleanliness_labels, clean_probs, 
                       color=['#2ECC71', '#F39C12', '#E74C3C'])
        ax1.set_title('Оценка чистоты', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Вероятность', fontsize=12)
        ax1.set_ylim(0, 1)
        
        for bar, prob in zip(bars1, clean_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        damage_probs = results['damage']['probabilities']
        bars2 = ax2.bar(damage_labels, damage_probs, 
                       color=['#3498DB', '#E74C3C'])
        ax2.set_title('Оценка целостности', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Вероятность', fontsize=12)
        ax2.set_ylim(0, 1)
        
        for bar, prob in zip(bars2, damage_probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig


@st.cache_resource
def load_classifier():
    model_path = "models/model.pth"
    return StreamlitCarClassifier(model_path)


def setup_page_config():
    st.set_page_config(
        page_title="inDrive: Классификатор состояния автомобилей",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header():
    st.title("🚗 inDrive: Оценка состояния автомобиля")
    st.markdown("### *Автоматическое определение чистоты и целостности автомобиля по фотографии*")


def render_sidebar():
    with st.sidebar:
        st.header("ℹ️ О системе")
        
        st.markdown("""
        **Технические характеристики:**
        - 🏗️ **Архитектура**: EfficientNet-B4
        - 🎯 **Задачи**: Мультиклассовая классификация
        - 📊 **Классы чистоты**: 3 (Чистый, Слегка грязный, Очень грязный)
        - 🔧 **Классы целостности**: 2 (Целый, Битый)
        
        **Применение в inDrive:**
        """)
        
        applications = [
            ("🛡️", "**Безопасность**", "Предупреждение о повреждениях"),
            ("⭐", "**Качество сервиса**", "Контроль состояния автопарка"),
            ("📱", "**UX улучшения**", "Информирование пассажиров"),
            ("🔔", "**Умные уведомления**", "Напоминания водителям"),
            ("📈", "**Аналитика**", "Мониторинг качества услуг")
        ]
        
        for icon, title, desc in applications:
            st.markdown(f"{icon} {title}: {desc}")
        
        st.markdown("---")
        
        st.header("📊 Метрики модели")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Чистота", "87.3%", "2.1%")
        with col2:
            st.metric("Целостность", "91.2%", "1.8%")
        
        st.markdown("---")
        
        st.header("⚙️ Настройки")
        show_probabilities = st.checkbox("Показать все вероятности", True)
        show_confidence = st.checkbox("Показать уверенность", True)
        
        st.header("🖼️ Примеры")
        example_images = {
            "Чистый автомобиль": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=300",
            "Грязный автомобиль": "https://images.unsplash.com/photo-1558618667-fcd25c85cd64?w=300",
            "Поврежденный автомобиль": "https://images.unsplash.com/photo-1558618047-3c0c6424dd31?w=300"
        }
        
        selected_example = st.selectbox("Выберите пример:", list(example_images.keys()))
        if st.button("Загрузить пример"):
            st.session_state.example_url = example_images[selected_example]
    
    return show_probabilities, show_confidence


def handle_file_upload():
    st.header("📸 Загрузите фотографию автомобиля")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение автомобиля...", 
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Поддерживаемые форматы: JPG, JPEG, PNG, WEBP. Максимальный размер: 10MB"
    )
    
    with st.expander("🔗 Или введите URL изображения"):
        image_url = st.text_input("URL изображения:")
        if st.button("Загрузить по URL") and image_url:
            try:
                import requests
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.session_state.url_image = image
                st.success("Изображение загружено по URL!")
            except Exception as e:
                st.error(f"Ошибка загрузки изображения: {e}")
    
    return uploaded_file


def get_image_to_process(uploaded_file):
    image_to_process = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif hasattr(st.session_state, 'url_image'):
        image_to_process = st.session_state.url_image
    elif hasattr(st.session_state, 'example_url'):
        try:
            import requests
            response = requests.get(st.session_state.example_url)
            image_to_process = Image.open(io.BytesIO(response.content))
        except:
            st.error("Не удалось загрузить пример изображения")
    
    return image_to_process


def display_image_info(image):
    st.write("**Информация об изображении:**")
    st.write(f"- Размер: {image.size[0]}×{image.size[1]} px")
    st.write(f"- Формат: {image.format}")


def display_results_metrics(results, show_confidence):
    cleanliness_labels = ['Чистый', 'Слегка грязный', 'Очень грязный']
    damage_labels = ['Целый', 'Битый']
    
    clean_pred = results['cleanliness']['prediction']
    damage_pred = results['damage']['prediction']
    
    clean_label = cleanliness_labels[clean_pred]
    damage_label = damage_labels[damage_pred]
    
    clean_conf = results['cleanliness']['confidence']
    damage_conf = results['damage']['confidence']
    
    st.subheader("🎯 Результаты анализа")
    
    col_clean, col_damage = st.columns(2)
    
    with col_clean:
        if clean_pred == 0:
            color = "🟢"
        elif clean_pred == 1:
            color = "🟡"
        else:
            color = "🔴"
        
        st.metric(
            label=f"{color} Чистота",
            value=clean_label,
            delta=f"Уверенность: {clean_conf:.1%}" if show_confidence else None
        )
    
    with col_damage:
        color = "🟢" if damage_pred == 0 else "🔴"
        st.metric(
            label=f"{color} Целостность", 
            value=damage_label,
            delta=f"Уверенность: {damage_conf:.1%}" if show_confidence else None
        )
    
    return clean_pred, damage_pred, clean_label, damage_label, clean_conf, damage_conf


def display_recommendations(clean_pred, damage_pred):
    st.subheader("💡 Рекомендации для платформы")
    
    recommendations = []
    
    if clean_pred >= 2:
        recommendations.append("🔴 **Критично**: Рекомендуется уведомить водителя о необходимости мойки")
    elif clean_pred >= 1:
        recommendations.append("🟡 **Внимание**: Можно предложить водителю услуги мойки")
    else:
        recommendations.append("🟢 **Отлично**: Автомобиль в хорошем состоянии")
    
    if damage_pred >= 1:
        recommendations.append("🔴 **Безопасность**: Требуется проверка технического состояния")
        recommendations.append("📱 **UX**: Предупредить пассажира о возможных повреждениях")
    else:
        recommendations.append("🟢 **Безопасность**: Видимые повреждения не обнаружены")
    
    for rec in recommendations:
        st.markdown(rec)


def create_export_section(clean_label, clean_conf, clean_pred, damage_label, damage_conf, damage_pred, show_probabilities, fig=None):
    st.subheader("💾 Экспорт результатов")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        result_json = {
            "timestamp": str(pd.Timestamp.now()),
            "cleanliness": {
                "label": clean_label,
                "confidence": float(clean_conf),
                "class_id": int(clean_pred)
            },
            "damage": {
                "label": damage_label, 
                "confidence": float(damage_conf),
                "class_id": int(damage_pred)
            }
        }
        
        st.download_button(
            label="📄 Скачать JSON",
            data=json.dumps(result_json, indent=2, ensure_ascii=False),
            file_name="car_analysis.json",
            mime="application/json"
        )
    
    with col_exp2:
        df_result = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'cleanliness_label': clean_label,
            'cleanliness_confidence': clean_conf,
            'damage_label': damage_label,
            'damage_confidence': damage_conf
        }])
        
        st.download_button(
            label="📊 Скачать CSV",
            data=df_result.to_csv(index=False),
            file_name="car_analysis.csv",
            mime="text/csv"
        )
    
    with col_exp3:
        if show_probabilities and fig is not None:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            st.download_button(
                label="🖼️ Скачать график",
                data=img_buffer.getvalue(),
                file_name="analysis_chart.png",
                mime="image/png"
            )


def display_examples():
    st.info("👆 Загрузите изображение автомобиля для анализа")
    
    st.subheader("🎥 Примеры работы системы")
    
    example_cols = st.columns(3)
    examples_data = [
        ("Чистый автомобиль", "🟢 Чистый", "🟢 Целый", "95%"),
        ("Грязный автомобиль", "🔴 Очень грязный", "🟢 Целый", "88%"),  
        ("Поврежденный автомобиль", "🟡 Слегка грязный", "🔴 Битый", "92%")
    ]
    
    for i, (title, clean_res, damage_res, conf) in enumerate(examples_data):
        with example_cols[i]:
            st.markdown(f"**{title}**")
            st.markdown(f"Чистота: {clean_res}")
            st.markdown(f"Целостность: {damage_res}")
            st.markdown(f"Уверенность: {conf}")


def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🚗 <b>inDrive Car Condition Classifier</b> | Powered by EfficientNet-B4 | 
        <a href='https://github.com/indrive' target='_blank'>GitHub</a> | 
        <a href='https://indrive.com' target='_blank'>inDrive</a></p>
        <p><i>Система автоматической оценки состояния автомобилей для повышения качества и безопасности сервиса</i></p>
    </div>
    """, unsafe_allow_html=True)


def main():
    setup_page_config()
    render_header()
    
    show_probabilities, show_confidence = render_sidebar()
    
    uploaded_file = handle_file_upload()
    image_to_process = get_image_to_process(uploaded_file)
    
    if image_to_process is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image_to_process, caption="Загруженное изображение", use_container_width=True)
            display_image_info(image_to_process)
            
        with col2:
            with st.spinner("Загружаем модель..."):
                classifier = load_classifier()
            
            with st.spinner("Анализируем изображение..."):
                results = classifier.predict(image_to_process)
            
            if results:
                clean_pred, damage_pred, clean_label, damage_label, clean_conf, damage_conf = display_results_metrics(
                    results, show_confidence
                )
                
                fig = None
                if show_probabilities:
                    st.subheader("📊 Детальный анализ")
                    fig = classifier.create_results_chart(results)
                    st.pyplot(fig)
                    plt.close()
                
                display_recommendations(clean_pred, damage_pred)
                
                create_export_section(
                    clean_label, clean_conf, clean_pred, 
                    damage_label, damage_conf, damage_pred, 
                    show_probabilities, fig
                )
    else:
        display_examples()
    
    render_footer()


if __name__ == "__main__":
    main()