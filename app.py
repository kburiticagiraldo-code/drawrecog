import base64
import io
import os

import numpy as np
import streamlit as st
from openai import OpenAI
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODELO = "gpt-4o-mini"  # si lee mal tu letra, prueba con "gpt-4o"

PROMPT = (
    "La imagen contiene una ecuación o ejercicio de matemáticas escrito a mano. "
    "Responde en español y en este formato (usa LaTeX con $$...$$ para las fórmulas):\n"
    "**Ecuación detectada:** transcríbela.\n"
    "**Solución paso a paso:** pasos numerados y cortos.\n"
    "**Resultado:** el resultado final.\n"
    "Si no se entiende lo escrito, dilo y pide que lo dibuje más grande y claro."
)

# Streamlit
st.set_page_config(page_title="Ecuaciones a mano", page_icon="🧮")
st.title("Ecuaciones a mano")
with st.sidebar:
    st.subheader("Acerca de:")
    st.subheader("Escribe una ecuación o ejercicio en el panel y la IA lo resuelve paso a paso.")
    stroke_width = st.slider("Selecciona el ancho de línea", 1, 30, 4)
st.subheader("Escribe la ecuación en el panel y presiona el botón para resolverla")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=250,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

# Clave: primero intenta st.secrets; si no hay, la pide en pantalla
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = st.text_input("Ingresa tu Clave", type="password")

analyze_button = st.button("Resolver ecuación", type="secondary")

if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Resolviendo ..."):
        # Imagen del canvas sobre fondo blanco, codificada en base64
        arr = np.array(canvas_result.image_data)
        img = Image.fromarray(arr.astype("uint8"), "RGBA")
        fondo = Image.new("RGB", img.size, "white")
        fondo.paste(img, mask=img.split()[3])
        buffer = io.BytesIO()
        fondo.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=MODELO,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=800,
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
elif analyze_button and not api_key:
    st.warning("Por favor ingresa tu API key.")
