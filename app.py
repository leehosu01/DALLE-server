import base64
import time
from io import BytesIO
from queue import Empty, Queue
from threading import Thread

import ruclip
from flask import Flask, jsonify, request
from PIL import Image
from rudalle import get_realesrgan, get_rudalle_model, get_tokenizer, get_vae
from rudalle.pipelines import cherry_pick_by_ruclip, generate_images, super_resolution
from translators import bing as translate_api

app = Flask(__name__)

requests_queue = Queue()  # request queue.
REQUEST_BATCH_SIZE = 4  # max request size.
CHECK_INTERVAL = 0.1

# load model
batch_size = 8
device = "cuda"
dalle = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device)
realesrgan = get_realesrgan("x2", device=device)  # x2/x4/x8
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)  # for stable generations you should use dwt=False
clip, processor = ruclip.load("ruclip-vit-base-patch32-384", device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=batch_size)

top_k = 512
top_p = 0.995


samples_for_one = 16


def handle_requests_by_batch():
    while True:
        requests = requests_queue.get()
        try:
            requests["output"] = make_images(requests["input"][0], requests["input"][1])

        except Exception as e:
            requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


def make_images(text_input, num_images):
    try:
        text = translate_api(text_input, from_language="auto", to_language="ja")
        text = translate_api(text, from_language="ja", to_language="ru")

        images_num = num_images * samples_for_one
        pil_images, ppl_scores = generate_images(
            text,
            tokenizer,
            dalle,
            vae,
            top_k=top_k,
            images_num=images_num,
            top_p=top_p,
            bs=batch_size,
        )
        # CLIP
        top_images = cherry_pick_by_ruclip(
            pil_images, text, clip_predictor, count=num_images
        )
        # Super Resolution
        sr_images = super_resolution(top_images, realesrgan)

        response = []

        for image in sr_images:
            img = Image.fromarray(image)

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            response.append(img_str)

        return response

    except Exception as e:
        print("Error occur in script generating!", e)
        return jsonify({"Error": e}), 500


@app.route("/generate", methods=["POST"])
def generate():
    if requests_queue.qsize() > REQUEST_BATCH_SIZE:
        return jsonify({"Error": "Too Many Requests. Please try again later"}), 429

    try:
        args = []
        json_data = request.get_json()
        text_input = json_data["text"]
        num_images = json_data["num_images"]

        if num_images > 10:
            return (
                jsonify(
                    {"Error": "Too many images requested. Request no more than 10"}
                ),
                500,
            )

        args.append(text_input)
        args.append(num_images)

    except Exception as e:
        return jsonify({"Error": "Invalid request"}), 500

    req = {"input": args}
    requests_queue.put(req)

    while "output" not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req["output"])


@app.route("/healthz", methods=["GET"])
def health_check():
    return "Health", 200


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
