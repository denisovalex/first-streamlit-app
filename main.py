import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

MODEL_ID = "runwayml/stable-diffusion-v1-5"


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def init_pipe():
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
        pipe = pipe.to("cpu")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    return pipe


def generate_images(prompt, num_images, num_inference_steps, guidance_scale):
    pipe = init_pipe()
    return pipe([prompt] * num_images, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images


def start_streamlit():
    st.title("Генератор картинок по тексту")

    text_input = st.text_input("Напишите здесь какую катинку вы хотите сгенерировать")

    num_inference_steps = st.slider("Число шагов (влияет на скорость работы)",
                                    value=50, min_value=1, max_value=800, step=1)

    guidance_scale = st.slider("Guidance scale (чем больше значение, тем больше соответствие тексту)",
                               value=6.0, min_value=1.1, max_value=15.0, step=0.1)

    num_images = st.slider("Укажите сколько картинок вы хотите сгенерировать",
                           value=1, min_value=1, max_value=5, step=1)

    if st.button("Сгенерировать!"):

        if text_input != "":

            images = generate_images(text_input, num_images, num_inference_steps, guidance_scale)

            for image in images:
                st.image(image)

            if st.button("Скачать всё"):
                for idx, image in images:
                    image.save(f"image-{idx}.png")

        else:
            st.text("Вы не ввели текст. Напишите, какую картинку вы бы хотели получить, и поробуйте снова!")


if __name__ == '__main__':
    start_streamlit()
