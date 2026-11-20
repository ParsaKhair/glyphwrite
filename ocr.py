import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Punctuation written against the word before it, and against the word after it
CLOSERS = set(",.;:!?)]}")
OPENERS = set("([{")


# Loads TrOCR. Returns recognize(images) -> [(text, conf, pieces)]: conf = geometric-mean token probability,
# pieces = [(token text, log-probability)] for the text's tokens in order.
def load_trocr(model_name="microsoft/trocr-base-handwritten", batch_size=16):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
    pad = model.generation_config.pad_token_id
    special = set(processor.tokenizer.all_special_ids)

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
            texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)

            # Pairs each text token with its log-probability, dropping start, end and padding tokens
            for seq, lps, text, c in zip(gen.sequences[:, 1:].tolist(), logp.tolist(), texts, conf.tolist()):
                pieces = [(processor.tokenizer.decode([t]), lp) for t, lp in zip(seq, lps) if t not in special]
                out.append((text, c, pieces))
        return out

    return recognize


# Groups OCR tokens into words with 1 confidence each. TrOCR spaces out punctuation ("mixing , then"),
# so closing punctuation joins the word before it and opening punctuation the word after it.
# Takes [(token text, log-probability)], returns [(word, conf)] with conf = geometric-mean token probability.
def ocr_words(pieces):
    words = []
    space = glue = False
    for piece, lp in pieces:
        token = piece.strip()
        # Remembers a space-only token as a word break
        if not token:
            space = True
            continue
        # Starts a new word after a space, unless the token closes the last word or the last word was an opener
        if not words or ((space or piece[0].isspace()) and not glue and not set(token) <= CLOSERS):
            words.append(["", []])
        words[-1][0] += token
        words[-1][1].append(lp)
        space, glue = False, set(token) <= OPENERS
    return [(word, float(np.exp(np.mean(lps)))) for word, lps in words]


# Renders a whole line as dark ink on white for OCR. Takes a Line, returns a PIL image.
def line_image(line, pad=0.3):
    ink = np.pad(line.density, int(pad * line.xheight_px))
    return Image.fromarray((255 * (1 - ink)).astype(np.uint8))


# OCRs every line in the corpus. Stores its words joined by single spaces, its confidence, and 1 confidence per word.
def read_lines(corpus, recognize):
    lines = list(corpus.lines.values())
    for line, (_, conf, pieces) in zip(lines, recognize([line_image(line) for line in lines])):
        words = ocr_words(pieces)
        line.text = " ".join(word for word, _ in words)
        line.ocr_conf = float(conf)
        line.word_confs = [c for _, c in words]


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
    for seg, (text, conf, _) in zip(segs, recognize(images)):
        seg.text = "".join(text.split())
        seg.ocr_conf = float(conf)


# Text of 1 line, built from its segments in order.
def line_text(corpus, line):
    return " ".join(corpus.segments[i].text for i in line.segment_ids)
