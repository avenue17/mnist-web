import os
import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, iris.target_names, accuracy


model, target_names, test_accuracy = train_model()


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    probabilities = model.predict_proba(data)[0]

    result = {
        target_names[i]: float(probabilities[i])
        for i in range(len(target_names))
    }

    predicted_index = int(np.argmax(probabilities))
    predicted_name = target_names[predicted_index]

    message = f"예측 결과: {predicted_name}\n테스트 정확도: {test_accuracy:.3f}"
    return message, result


with gr.Blocks(title="Iris Flower Classifier") as app:
    gr.Markdown("# Iris Flower Classifier")
    gr.Markdown("꽃받침과 꽃잎의 길이, 너비를 입력하면 붓꽃의 품종을 예측합니다.")

    with gr.Row():
        with gr.Column():
            sepal_length = gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Sepal length")
            sepal_width = gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Sepal width")
            petal_length = gr.Slider(1.0, 7.0, value=1.4, step=0.1, label="Petal length")
            petal_width = gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Petal width")
            button = gr.Button("Predict")

        with gr.Column():
            output_text = gr.Textbox(label="Result")
            output_label = gr.Label(label="Prediction probabilities")

    button.click(
        fn=predict_iris,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=[output_text, output_label]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )