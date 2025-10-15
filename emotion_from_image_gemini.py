#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detecta emocion facial con Gemini a partir de:
- una imagen local (--image),
- una URL (--url), o
- la camara (--camera) con captura al presionar 'p'.

Uso:
  python emotion_from_image_gemini.py --image descarga.jpg
  python emotion_from_image_gemini.py --url https://ejemplo.com/rostro.jpg
  python emotion_from_image_gemini.py --camera  # Abre camara, 'p' foto, 'q' salir

Requiere:
  pip install google-generativeai pillow requests
  (para camara) pip install opencv-python
  (opcional en Raspberry Pi) sudo apt install python3-picamera2

Y definir la variable de entorno:
  GOOGLE_API_KEY
"""

import os
import io
import json
import argparse
import base64
from typing import Dict, Any, Tuple

# Dependencias
try:
    import google.generativeai as genai
except ImportError as e:
    raise SystemExit("Falta google-generativeai. Instala con: pip install google-generativeai") from e

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Falta Pillow. Instala con: pip install pillow")

try:
    import requests
except ImportError:
    raise SystemExit("Falta requests. Instala con: pip install requests")

try:
    import cv2  # Para camara y captura
except ImportError:
    cv2 = None  # Permitimos usar --image/--url sin OpenCV

try:
    from picamera2 import Picamera2  # Fallback para Raspberry Pi (libcamera)
    _PICAMERA2_AVAILABLE = True
except Exception:
    Picamera2 = None
    _PICAMERA2_AVAILABLE = False


EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]

SYSTEM_INSTRUCTION = """Eres un clasificador de emociones faciales estricto.
Analiza SOLO la expresion facial visible en la imagen y devuelve un JSON VALIDO con este formato:

{
  "emotion": "<una de: neutral, happy, sad, angry, fear, surprise, disgust>",
  "confidence": <0..1>,
  "probabilities": {
    "neutral": <0..1>,
    "happy": <0..1>,
    "sad": <0..1>,
    "angry": <0..1>,
    "fear": <0..1>,
    "surprise": <0..1>,
    "disgust": <0..1>
  },
  "notes": "<breve justificacion en una frase>"
}

Reglas:
- La suma de probabilities ~= 1.0.
- "emotion" debe ser la etiqueta con mayor probabilidad.
- Si no hay rostro o esta muy tapado, responde "neutral" con baja confidence y en "notes" indica la limitacion.
- No incluyas texto fuera del JSON. No uses Markdown.
"""

USER_PROMPT = """Clasifica la emocion facial. Atiende a cejas, ojos, boca y tensiones musculares.
Devuelve unicamente el JSON solicitado, nada mas.
"""


def load_image_from_path(path: str) -> Tuple[bytes, str]:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    # Guardamos como JPEG para tamano razonable
    img.save(buf, format="JPEG", quality=92)
    data = buf.getvalue()
    return data, "image/jpeg"


def load_image_from_url(url: str) -> Tuple[bytes, str]:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        # Intento de normalizacion a JPEG si no viene tipo util
        content_type = "image/jpeg"
    return resp.content, content_type


def ensure_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("No se encontro GOOGLE_API_KEY en variables de entorno.")
    genai.configure(api_key=api_key)


def build_model(model_name: str = "gemini-2.0-flash") -> genai.GenerativeModel:
    # Puedes cambiar a "gemini-2.0-pro" si quieres mas calidad (mas costo/latencia).
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "response_mime_type": "application/json",  # pedimos JSON directo
        }
    )


def call_gemini(model: genai.GenerativeModel, image_bytes: bytes, mime: str) -> Dict[str, Any]:
    # Construimos el "content" multimodal: prompt + imagen
    parts = [
        {"text": USER_PROMPT},
        {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        },
    ]

    response = model.generate_content(parts)
    raw = response.text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raw_fixed = raw.strip().strip("` \n")
        data = json.loads(raw_fixed)

    return data


def validate_and_normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    # Chequeos basicos y normalizacion de probabilidades
    probs = result.get("probabilities", {})
    # Garantizamos que existan todas las etiquetas
    probs_complete = {}
    total = 0.0
    for lab in EMOTION_LABELS:
        val = float(probs.get(lab, 0.0))
        if val < 0:
            val = 0.0
        probs_complete[lab] = val
        total += val

    # Re-normalizar si es necesario
    if total > 0:
        probs_complete = {k: v / total for k, v in probs_complete.items()}
    else:
        # Si todo es 0, ponemos distribucion casi uniforme con pequeno sesgo neutral
        uniform = 1.0 / len(EMOTION_LABELS)
        probs_complete = {k: uniform for k in EMOTION_LABELS}
        probs_complete["neutral"] += 0.001
        s = sum(probs_complete.values())
        probs_complete = {k: v / s for k, v in probs_complete.items()}

    # Emocion = argmax
    emotion = max(probs_complete.items(), key=lambda kv: kv[1])[0]
    confidence = float(result.get("confidence", probs_complete[emotion]))

    normalized = {
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probs_complete.items()},
        "notes": result.get("notes", "")
    }
    return normalized


def pretty_print(res: Dict[str, Any]):
    print(json.dumps(res, ensure_ascii=False, indent=2))


def run_classification_with_path(model: genai.GenerativeModel, image_path: str):
    image_bytes, mime = load_image_from_path(image_path)
    raw = call_gemini(model, image_bytes, mime)
    result = validate_and_normalize(raw)
    pretty_print(result)


def camera_loop_and_classify(
    temp_filename: str = "captura_temp.jpg",
    cam_index: int = 0,
    cam_width: int = 640,
    cam_height: int = 480,
):
    if cv2 is None and not _PICAMERA2_AVAILABLE:
        raise SystemExit("Falta opencv-python o picamera2. Instala con: pip install opencv-python o sudo apt install python3-picamera2")

    ensure_api_key()
    model = build_model("gemini-2.0-flash")

    temp_path = os.path.join(os.getcwd(), temp_filename)

    cap = None
    use_picamera2 = False
    picam = None

    # Intento OpenCV (preferido por simplicidad y vista previa)
    if cv2 is not None:
        backend = getattr(cv2, "CAP_V4L2", 0)
        try:
            cap = cv2.VideoCapture(cam_index, backend) if backend else cv2.VideoCapture(cam_index)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        except Exception:
            cap = None

    # Fallback a Picamera2 si OpenCV no abre (util para camara CSI/libcamera)
    if (cap is None or not getattr(cap, 'isOpened', lambda: False)()) and _PICAMERA2_AVAILABLE and cv2 is not None:
        try:
            picam = Picamera2()
            cfg = picam.create_preview_configuration(main={"size": (cam_width, cam_height), "format": "RGB888"})
            picam.configure(cfg)
            picam.start()
            use_picamera2 = True
        except Exception:
            picam = None

    if (cap is None or not getattr(cap, 'isOpened', lambda: False)()) and not use_picamera2:
        raise SystemExit("No se pudo abrir la camara. En Raspberry Pi, prueba instalar 'python3-picamera2' o usa una camara USB compatible.")

    print("Camara lista. Ventana activa:\n - Presiona 'p' para tomar foto y clasificar.\n - Presiona 'q' para salir.")

    try:
        while True:
            if use_picamera2:
                frame = picam.capture_array()
                ok = frame is not None
            else:
                ok, frame = cap.read()

            if not ok:
                print("No se pudo leer de la camara.")
                break

            if cv2 is None:
                print("OpenCV no disponible para vista previa. Pulsa Ctrl+C para terminar.")
                break
            else:
                cv2.imshow("Camara - 'p' foto, 'q' salir", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                if key == ord('p'):
                    ok_save = cv2.imwrite(temp_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if not ok_save:
                        print("No se pudo guardar la captura.")
                        continue

                    print(f"Captura guardada: {temp_path}")
                    print(f"Ejecutando: python emotion_from_image_gemini.py --image {temp_path}")
                    try:
                        run_classification_with_path(model, temp_path)
                    except Exception as e:
                        print(f"Error al clasificar: {e}")
                    print("Listo. Presiona 'p' de nuevo o 'q' para salir.")
    finally:
        try:
            if cap is not None:
                cap.release()
        finally:
            if cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if use_picamera2 and picam is not None:
                try:
                    picam.stop()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Clasificacion de emociones con Gemini (imagen, URL o camara).")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--image", type=str, help="Ruta a imagen local (jpg/png).")
    src.add_argument("--url", type=str, help="URL directa a una imagen.")
    src.add_argument("--camera", action="store_true", help="Usar camara: 'p' para foto, 'q' para salir.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Nombre de modelo (por ej., gemini-2.0-flash o gemini-2.0-pro).")
    parser.add_argument("--cam-index", type=int, default=0, help="Indice de camara (V4L2). Por defecto 0.")
    parser.add_argument("--cam-width", type=int, default=640, help="Ancho de captura (px). Por defecto 640.")
    parser.add_argument("--cam-height", type=int, default=480, help="Alto de captura (px). Por defecto 480.")
    parser.add_argument("--temp-name", type=str, default="captura_temp.jpg", help="Nombre de archivo temporal para capturas.")
    args = parser.parse_args()

    # Si se pide camara, o si no se especifica --image/--url, abrimos la camara.
    if args.camera or (not args.image and not args.url):
        camera_loop_and_classify(
            temp_filename=args.temp_name,
            cam_index=args.cam_index,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
        )
        return

    ensure_api_key()
    model = build_model(args.model)

    if args.image:
        image_bytes, mime = load_image_from_path(args.image)
    else:
        image_bytes, mime = load_image_from_url(args.url)

    raw = call_gemini(model, image_bytes, mime)
    result = validate_and_normalize(raw)
    pretty_print(result)


if __name__ == "__main__":
    main()

