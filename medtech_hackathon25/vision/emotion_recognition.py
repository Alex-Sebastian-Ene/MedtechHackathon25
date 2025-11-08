"""PyTorch helpers for face recognition and emotion classification from static images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import time
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_EMOTION_LABELS: Tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

EMOTION_SCORES: Dict[str, int] = {
    "angry": 2,
    "disgust": 3,
    "fear": 3,
    "sad": 2,
    "neutral": 5,
    "surprise": 7,
    "happy": 9,
}


@dataclass
class DetectedFace:
    tensor: torch.Tensor
    box: np.ndarray
    probability: float


class FaceEncoder:
    """Produces face embeddings for identity matching."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(self.device)

    @torch.inference_mode()
    def encode(self, face: torch.Tensor) -> torch.Tensor:
        face = face.to(self.device)
        if face.ndim == 3:
            face = face.unsqueeze(0)
        embedding = self.model(face)
        return F.normalize(embedding, p=2, dim=-1)


class FaceRegistry:
    """Simple in-memory store mapping identities to mean embeddings."""

    def __init__(self, encoder: FaceEncoder, threshold: float = 0.6) -> None:
        self.encoder = encoder
        self.threshold = threshold
        self.embeddings: Dict[str, torch.Tensor] = {}

    def register(self, name: str, faces: Iterable[torch.Tensor]) -> None:
        vectors = [self.encoder.encode(face).mean(dim=0) for face in faces]
        if not vectors:
            raise ValueError("No face tensors provided for registration.")
        stacked = torch.stack(vectors)
        mean_vec = F.normalize(stacked.mean(dim=0), p=2, dim=0)
        self.embeddings[name] = mean_vec

    @torch.inference_mode()
    def identify(self, face: torch.Tensor) -> Tuple[str, float]:
        query = self.encoder.encode(face).squeeze(0)
        best_name = "unknown"
        best_score = 0.0
        for name, ref in self.embeddings.items():
            score = float(F.cosine_similarity(query, ref, dim=0))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < self.threshold:
            return "unknown", best_score
        return best_name, best_score


class PretrainedEmotionClassifier:
    """Uses a pretrained FER2013 model served via torch.hub."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "dima806/facial_emotions_image_detection",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_compiler = getattr(torch, "compiler", None)
        supports_fast = bool(getattr(torch_compiler, "is_compiling", None))
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=supports_fast)
        except Exception:
            # Fast processors are not available for every checkpoint; fall back silently.
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        try:
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - depends on network
            raise RuntimeError(
                "Unable to download pretrained emotion weights. Ensure internet access or "
                "provide a local checkpoint via emotion_checkpoint."
            ) from exc
        self.model.to(self.device).eval()
        self.labels: Tuple[str, ...] = tuple(
            str(self.model.config.id2label[idx]).lower() for idx in range(self.model.config.num_labels)
        )
        self.label_mapping = {label: label for label in self.labels}

    @torch.inference_mode()
    def predict(self, face_tensor: torch.Tensor) -> Tuple[str, float]:
        pil_face = transforms.ToPILImage()(face_tensor.cpu().clamp(0, 1))
        inputs = self.processor(images=pil_face, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        score, idx = probs.squeeze(0).max(dim=0)
        label = self.labels[int(idx)]
        return label, float(score)


class FaceEmotionRecognizer:
    """High-level facade combining detection, identification, and emotion scoring."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        emotion_checkpoint: Optional[Path] = None,
        detection_image_size: int = 224,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(
            image_size=detection_image_size,
            margin=20,
            post_process=True,
            device=self.device,
            keep_all=True,
        )
        self.encoder = FaceEncoder(device=self.device)
        self.registry = FaceRegistry(self.encoder)
        if emotion_checkpoint:
            self.emotion_model = self._load_custom_emotion_model(emotion_checkpoint)
        else:
            self.emotion_model = PretrainedEmotionClassifier(device=self.device)
    def _load_custom_emotion_model(self, path: Path) -> PretrainedEmotionClassifier:
        return PretrainedEmotionClassifier(device=self.device, model_name=str(path))

    @torch.inference_mode()
    def _detect_faces(self, image: Image.Image) -> List[DetectedFace]:
        boxes, probs = self.detector.detect(image)
        if boxes is None or probs is None:
            return []
        faces = self.detector.extract(image, boxes, save_path=None)
        if isinstance(faces, torch.Tensor):
            if faces.ndim == 3:
                faces = faces.unsqueeze(0)
            face_iterable = [face.detach().cpu() for face in faces]
        else:
            face_iterable = [torch.tensor(face).permute(2, 0, 1) for face in faces]

        detections: List[DetectedFace] = []
        for face_tensor, box, prob in zip(face_iterable, boxes, probs):
            score = float(prob) if prob is not None else 0.0
            if score > 0.85:
                detections.append(
                    DetectedFace(
                        tensor=face_tensor,
                        box=np.array(box, dtype=float),
                        probability=score,
                    )
                )
        return detections

    @torch.inference_mode()
    def _classify_emotion(self, face_tensor: torch.Tensor) -> Tuple[str, float]:
        return self.emotion_model.predict(face_tensor)

    @torch.inference_mode()
    def analyze(self, image_path: Path) -> List[Dict[str, object]]:
        image = Image.open(image_path).convert("RGB")
        return self.analyze_image(image)

    @torch.inference_mode()
    def analyze_image(self, image: Image.Image) -> List[Dict[str, object]]:
        detections = self._detect_faces(image)
        results: List[Dict[str, object]] = []
        for detected in detections:
            face_tensor = detected.tensor
            identity, confidence = self.registry.identify(face_tensor)
            emotion, emotion_score = self._classify_emotion(face_tensor)
            results.append(
                {
                    "identity": identity,
                    "identity_confidence": confidence,
                    "emotion": emotion,
                    "emotion_confidence": emotion_score,
                    "box": detected.box.tolist(),
                    "detection_confidence": detected.probability,
                }
            )
        return results

    @torch.inference_mode()
    def analyze_frame(self, frame: np.ndarray) -> List[Dict[str, object]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.analyze_image(image)

    def register_identity(self, name: str, image_paths: Iterable[Path]) -> None:
        tensors: List[torch.Tensor] = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            faces = self._detect_faces(image)
            tensors.extend(face.tensor for face in faces)
        if not tensors:
            raise RuntimeError(f"No faces detected for identity '{name}'.")
        self.registry.register(name, tensors)


def score_results(results: List[Dict[str, object]]) -> int:
    if not results:
        return 5
    total = 0.0
    weight = 0.0
    for item in results:
        emotion = str(item.get("emotion", "neutral")).lower()
        confidence = float(item.get("emotion_confidence", 1.0))
        score = EMOTION_SCORES.get(emotion, 5)
        total += score * confidence
        weight += confidence
    average = total / weight if weight else 5.0
    return max(1, min(10, round(average)))


def capture_and_rate(camera_index: int = 0) -> Tuple[List[Dict[str, object]], int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizer = FaceEmotionRecognizer(device=device)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera. Check that a webcam is available.")
    ok, frame = camera.read()
    camera.release()
    if not ok:
        raise RuntimeError("Failed to read frame from camera.")
    findings = recognizer.analyze_frame(frame)
    mood = score_results(findings)
    return findings, mood


def draw_detections(frame: np.ndarray, detections: List[Dict[str, object]], mood: int) -> np.ndarray:
    overlay = frame.copy()
    cv2.putText(
        overlay,
        f"Mood: {mood}/10",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    for item in detections:
        box = item.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        emotion = item.get("emotion", "?")
        emotion_conf = item.get("emotion_confidence", 0.0)
        identity = item.get("identity", "unknown")
        identity_conf = item.get("identity_confidence", 0.0)
        label = f"{identity} ({identity_conf:.2f})"
        emotion_label = f"{emotion} ({emotion_conf:.2f})"
        cv2.putText(overlay, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, emotion_label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return overlay


def monitor_camera(camera_index: int = 0, interval: float = 0.0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizer = FaceEmotionRecognizer(device=device)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera. Check that a webcam is available.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[monitor] Failed to read frame; retrying...")
                time.sleep(interval)
                continue
            detections = recognizer.analyze_frame(frame)
            mood = score_results(detections)
            annotated = draw_detections(frame, detections, mood)
            cv2.imshow("Emotion Monitor", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if interval > 0:
                time.sleep(interval)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _demo() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizer = FaceEmotionRecognizer(device=device)
    # recognizer.register_identity("alex", [Path("data/alex_1.jpg"), Path("data/alex_2.jpg")])
    sample_image = Path("data/example.jpg")
    if not sample_image.exists():
        print("Place a sample image at data/example.jpg to run the demo.")
        return
    image = cv2.imread(str(sample_image))
    if image is None:
        print("Unable to load sample image.")
        return
    findings = recognizer.analyze_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    mood = score_results(findings)
    overlay = draw_detections(image, findings, mood)
    for idx, item in enumerate(findings, start=1):
        print(f"Face {idx} -> {item['identity']} ({item['identity_confidence']:.2f})")
        print(f"             emotion={item['emotion']} ({item['emotion_confidence']:.2f})")
    print(f"Mood rating: {mood}/10")
    cv2.imshow("Emotion Monitor (demo)", overlay)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyWindow("Emotion Monitor (demo)")


if __name__ == "__main__":
    try:
        print("Starting continuous mood monitoring. Press Ctrl+C to stop.")
        monitor_camera()
    except RuntimeError as err:
        print(f"Live capture unavailable: {err}")
        _demo()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)
