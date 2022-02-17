import base64
import time
from io import BytesIO
from queue import Empty, Queue
from threading import Thread

import clip
import numpy as np
import torch
from dalle_pytorch import DALLE, VQGanVAE
from dalle_pytorch.tokenizer import tokenizer
from einops import repeat
from flask import Flask, jsonify, request
from PIL import Image
import tqdm


app = Flask(__name__)

requests_queue = Queue()  # request queue.
REQUEST_BATCH_SIZE = 4  # max request size.
CHECK_INTERVAL = 0.1

# load model
device = "cuda"
clip_model, clip_preprocess = clip.load("ViT-B/32", device)

vae = VQGanVAE(None, None)

load_obj = torch.load(
    "./16L_64HD_8H_512I_128T_cc12m_cc3m_3E.pt"
)  # model checkpoint : https://github.com/robvanvolt/DALLE-models/tree/main/models/taming_transformer
dalle_params, _, weights = (
    load_obj.pop("hparams"),
    load_obj.pop("vae_params"),
    load_obj.pop("weights"),
)
dalle_params.pop("vae", None)  # cleanup later

dalle = DALLE(vae=vae, **dalle_params).to(device)

dalle.load_state_dict(weights)

batch_size = 4
samples_for_one = 16

top_k = 0.9

# generate images

image_size = vae.image_size


def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= REQUEST_BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    requests["output"] = make_images(
                        requests["input"][0], requests["input"][1]
                    )

                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


def make_images(text_input, num_images):
    try:
        require_samples = samples_for_one * num_images

        text = tokenizer.tokenize([text_input], dalle.text_seq_len).to(device)
        text_chunk = repeat(text, "() n -> b n", b=batch_size)

        outputs = []
        for step_index in tqdm.trange(
            require_samples / batch_size, desc=f"generating images for - {text}"
        ):
            output = dalle.generate_images(text_chunk, filter_thres=top_k)
            outputs.append(output)
        outputs = torch.cat(outputs)

        clip_image_inputs = clip_preprocess(outputs).to(device)
        clip_text_inputs = clip.tokenize([text_input]).to(device)

        image_features = clip_model.encode_image(clip_image_inputs)
        text_features = clip_model.encode_text(clip_text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)  # [B, C]
        text_features /= text_features.norm(dim=-1, keepdim=True)  # [1, C]

        score = (image_features @ text_features.t()).squeeze(-1)
        arg_order = torch.argsort(score)
        outputs, score = outputs[arg_order], score[arg_order]

        response = []

        for i, image in tqdm.tqdm(
            enumerate(outputs[:num_images]), desc="saving images"
        ):
            np_image = np.moveaxis(image.cpu().numpy(), 0, -1)
            formatted = (np_image * 255).astype("uint8")

            img = Image.fromarray(formatted)

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            response.append(img_str)
        # TODO super resolution

        return (response, score)

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
