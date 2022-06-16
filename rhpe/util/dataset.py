import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from rhpe.util.transform import AffineTransformer, get_triangle
from typing_extensions import Self

MEAN_BBOX_WIDTH = 123
MEAN_BBOX_HEIGHT = 201


@dataclass
class COCOCropper:
    img_in_root: Path
    img_out_root: Path
    annot_in_path: Path
    annot_out_path: Path
    crop_range: tuple[float, float] = (0.2, 0.5)
    crop_prop: float = 1.0

    def create(self, size: int | None):
        annot_in_path = self.annot_in_path
        annot_out_path = self.annot_out_path
        img_in_root = self.img_in_root
        img_out_root = self.img_out_root
        with annot_in_path.open("r") as f:
            annot_full: dict = json.load(f)
        if not img_out_root.exists():
            img_out_root.mkdir(parents=True)
        if not annot_out_path.parent.exists():
            annot_out_path.parent.mkdir(parents=True)

        if isinstance(size, int):
            annotations = [
                Annotation(**dic) for dic in annot_full["annotations"][:size]
            ]
        else:
            annotations = [Annotation(**dic) for dic in annot_full["annotations"]]
        image_annotations = {
            dic["id"]: ImageAnnotation(**dic) for dic in annot_full["images"]
        }

        new_annotations = list()
        new_image_annotations = list()
        for idx, annotation in enumerate(annotations):
            file_name = f"{annotation.image_id:0>12}.jpg"
            img_path = img_in_root.joinpath(file_name)
            img = cv2.imread(str(img_path))
            new_annotation, new_img = crop_data(annotation, img, self.crop_range, idx)
            new_image_annotation = ImageAnnotation.construct(
                image_annotations[annotation.image_id], new_annotation, new_img
            )
            save_image(img_out_root.joinpath(file_name), new_img)
            new_annotations.append(asdict(new_annotation))
            new_image_annotations.append(asdict(new_image_annotation))

        annot_full["annotations"] = new_annotations
        annot_full["images"] = new_image_annotations
        with annot_out_path.open("w") as f:
            json.dump(annot_full, f)

    def save_cropped_image(self, bboxed_root: Path, size: int):
        with self.annot_out_path.open("r") as f:
            annot_full = json.load(f)

        annotations = [Annotation(**dic) for dic in annot_full["annotations"]]
        for annotation in annotations[:size]:
            filename = self.id_to_filename(annotation.image_id)
            path = self.img_out_root.joinpath(filename)
            img = cv2.imread(str(path))
            cropped_img = crop_image(img, annotation.bbox)
            sample_path = bboxed_root.joinpath(filename)
            cv2.imwrite(str(sample_path), cropped_img)

    def id_to_filename(self, idx: int) -> str:
        return f"{idx:0>12}.jpg"


@dataclass
class Annotation:
    segmentation: list[list[int]]
    num_keypoints: int
    area: float
    iscrowd: int
    keypoints: list[int]
    image_id: int
    bbox: tuple[float, float, float, float]
    category_id: int
    id: int

    def __post_init__(self):
        if isinstance(self.bbox, list):
            self.bbox = tuple(self.bbox)

    @property
    def joint_coverage(self) -> float:
        return self.num_keypoints / 17.0

    @property
    def bbox_coverage(self) -> float:
        xs = self.keypoints[0::3]
        ys = self.keypoints[1::3]
        return (
            np.sum([self.inside_bbox((x, y)) for x, y in zip(xs, ys)])
            / self.num_keypoints
        )

    @property
    def bbox_area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    def inside_bbox(self, keypoint: tuple[int, int]) -> bool:
        xmin, ymin, width, height = self.bbox
        xmax = xmin + width
        ymax = ymin + height
        x, y = keypoint
        return xmin <= x <= xmax and ymin <= y <= ymax


@dataclass
class ImageAnnotation:
    license: int
    file_name: str
    coco_url: str
    width: int
    height: int
    date_captured: str
    flickr_url: str
    id: int

    @classmethod
    def construct(
        cls, orig_img_annot: Self, new_annot: Annotation, new_img: np.ndarray
    ) -> Self:
        return cls(
            license=orig_img_annot.license,
            file_name=f"{new_annot.image_id:0>12}.jpg",
            coco_url="",
            width=new_img.shape[1],
            height=new_img.shape[0],
            date_captured=orig_img_annot.date_captured,
            flickr_url="",
            id=new_annot.image_id,
        )


def transform_keypoints(
    keypoints: list[int], transformer: AffineTransformer
) -> list[int]:
    x = keypoints[0::3]
    y = keypoints[1::3]
    v = np.array(keypoints[2::3])
    coordinates = np.stack([x, y], axis=1)
    new_coordinates = transformer.transform_position(coordinates)
    new_keypoints = np.concatenate([new_coordinates, v[:, np.newaxis]], axis=1)
    # TODO make (x, y) = (0, 0) for no annotations
    return new_keypoints.reshape((-1,)).tolist()


def generate_cropping_bbox(
    img: np.ndarray, crop_range: tuple[float, float]
) -> tuple[float, float, float, float]:
    """randomly generate cropping bbox
    for every bbox width and height, if its length is greater than its average,
    select cropping ratio p from [0, 0.5] and starting point s from [0, width - cropping length] unifromly.
    then return resulting bbox

    Args:
        img (np.ndarray): image to be cropped

    Returns:
        tuple[float, float, float, float]: generated bbox
    """
    height, width, _ = img.shape
    if width >= MEAN_BBOX_WIDTH:
        ratio = np.random.uniform(*crop_range)
        bbox_width = width * (1 - ratio)
        xmin = np.random.uniform(0, width - bbox_width)
    else:
        xmin = 0.0
        bbox_width = width
    if height >= MEAN_BBOX_HEIGHT:
        ratio = np.random.uniform(*crop_range)
        bbox_height = height * (1 - ratio)
        ymin = np.random.uniform(0, height - bbox_height)
    else:
        ymin = 0.0
        bbox_height = height
    return (xmin, ymin, bbox_width, bbox_height)


def crop_data(
    annotation: Annotation, img: np.ndarray, crop_range: tuple[float, float], idx: int
) -> tuple[Annotation, np.ndarray]:

    # get transformer
    xmin, ymin, width, height = annotation.bbox
    xmax = xmin + width
    ymax = ymin + height
    bbox = (xmin, ymin, xmax, ymax)
    size = (xmax - xmin, ymax - ymin)
    dst_bbox = (0, 0, *size)
    src = get_triangle(bbox)
    dst = get_triangle(dst_bbox)
    size = tuple(map(int, size))
    matrix = cv2.getAffineTransform(src, dst)
    transformer = AffineTransformer(matrix)

    # get new image and bbox
    new_img = transformer.transform_image(img, size)

    # get new bbox
    new_bbox = generate_cropping_bbox(new_img, crop_range)
    new_keypoints = transform_keypoints(annotation.keypoints, transformer)
    new_annotation = Annotation(
        annotation.segmentation,
        annotation.num_keypoints,
        annotation.area,
        annotation.iscrowd,
        new_keypoints,
        annotation.image_id,
        new_bbox,
        annotation.category_id,
        idx,
    )

    return new_annotation, new_img


def save_image(path: Path, img: np.ndarray):
    ret = cv2.imwrite(str(path), img)
    if not ret:
        raise ValueError("Failed to save image")


def crop_image(img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    bbox = (xmin, ymin, xmax, ymax)
    size = (xmax - xmin, ymax - ymin)
    dst_bbox = (0, 0, *size)
    src = get_triangle(bbox)
    dst = get_triangle(dst_bbox)
    size = tuple(map(int, size))
    matrix = cv2.getAffineTransform(src, dst)
    cropped_img = cv2.warpAffine(img, matrix, size, flags=cv2.INTER_LINEAR)
    return cropped_img


def inspect_dataset(coco_root: Path):
    with coco_root.joinpath("annotations", "person_keypoints_val2017.json").open(
        "r"
    ) as f:
        annot_full: dict = json.load(f)

    annotations = [Annotation(**dic) for dic in annot_full["annotations"][:20000]]

    print(
        f"Mean Joint Coverage: {np.mean([annotation.joint_coverage for annotation in annotations])}"
    )
    print(
        f"Mean BBox Coverage: {np.mean([annotation.bbox_coverage for annotation in annotations if annotation.num_keypoints != 0])}"
    )
    print(
        f"Mean BBox Area: {np.mean([annotation.bbox_area for annotation in annotations if annotation.num_keypoints != 0])}"
    )
    print(
        f"Mean Width: {np.mean([annotation.bbox[2] for annotation in annotations if annotation.num_keypoints != 0])}"
    )
    print(
        f"Mean Height: {np.mean([annotation.bbox[3] for annotation in annotations if annotation.num_keypoints != 0])}"
    )
