import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Loads TrOCR. Returns recognize(images) -> [(text, conf)], conf = geometric-mean token probability.
def load_trocr(model_name="microsoft/trocr-base-handwritten", batch_size=16):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
    pad = model.generation_config.pad_token_id

    @torch.no_grad()
    def recognize(images):
        out = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            pixels = processor(images=batch, return_tensors="pt").pixel_values
            gen = model.generate(pixels, max_new_tokens=32, output_scores=True, return_dict_in_generate=True)
            logp = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
            # Ignore padding after end-of-sequence
            keep = gen.sequences[:, 1:] != pad
            conf = ((logp * keep).sum(1) / keep.sum(1).clamp(min=1)).exp()
            out += zip(processor.batch_decode(gen.sequences, skip_special_tokens=True), conf.tolist())
        return out

    return recognize


# Renders 1 segment as dark ink on white for OCR. Takes its line + segment, returns a PIL image.
# Only the segment's own components are drawn; the stored density is untouched.
def segment_image(line, seg, pad=0.3):
    m = np.isin(line.labels, seg.comp_ids)
    rows, cols = np.nonzero(m)
    crop = (line.density * m)[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    crop = np.pad(crop, int(pad * line.xheight_px))
    return Image.fromarray((255 * (1 - crop)).astype(np.uint8))


# OCRs every segment in the corpus and stores text + confidence on it.
# Whitespace is stripped since segments are space-free by definition.
def read_segments(corpus, recognize):
    segs = list(corpus.segments.values())
    images = [segment_image(corpus.lines[s.line_id], s) for s in segs]
    for seg, (text, conf) in zip(segs, recognize(images)):
        seg.text = "".join(text.split())
        seg.ocr_conf = float(conf)


# Text of 1 line, built from its segments in order.
def line_text(corpus, line):
    return " ".join(corpus.segments[i].text for i in line.segment_ids)
