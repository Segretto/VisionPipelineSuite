import argparse
from pathlib import Path
import logging
import sys

# Import functions from the provided scripts
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_prep')))
from data_prep.split_data import split_data
from data_prep.coco2yolo import coco2yolo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ablation_datasets(main_dataset_path, num_chunks, output_folder, test_dataset_path, classes, mode="detection"):
    """
    Create ablation study datasets by splitting the main dataset into chunks
    and converting them into YOLO-compatible format.

    Args:
        main_dataset_path (str): Path to the main dataset folder.
        num_chunks (int): Number of chunks for ablation studies.
        output_folder (str): Path to the output folder for ablation datasets.
        test_dataset_path (str): Path to the test dataset folder.
        mode (str): Processing mode for YOLO conversion ('detection', 'segmentation', etc.).
    """
    main_dataset_path = Path(main_dataset_path)
    output_folder = Path(output_folder)
    test_dataset_path = Path(test_dataset_path)

    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Split the main dataset into chunks
    logger.info("Splitting the main dataset into ablation chunks...")
    split_data(
        images_dir=main_dataset_path / "images",
        coco_json_path=main_dataset_path / "annotations" / "coco.json",
        output_dir=output_folder,
        ablation=num_chunks,
        train_ratio=0.8,  # Not relevant for ablation mode
        val_ratio=0.1,    # Not relevant for ablation mode
        pose_estimation=False,
        rename_images=True,
        classes=classes,
    )

    # Convert each ablation chunk to YOLO format
    for ablation_chunk in output_folder.iterdir():
        if ablation_chunk.is_dir():
            logger.info(f"Converting ablation chunk to YOLO format: {ablation_chunk.name}")
            coco2yolo(
                dataset_path=str(ablation_chunk),
                dataset_splits=["train"],  # Ablation datasets typically focus on training
                mode=mode,
                ablation=True
            )

    # Optionally, convert the test dataset to YOLO format
    # if test_dataset_path.exists():
    #     logger.info("Converting the test dataset to YOLO format...")
    #     coco2yolo_main(
    #         dataset_path=str(test_dataset_path),
    #         dataset_splits=["test"],
    #         mode=mode
    #     )

    logger.info("Ablation study datasets created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ablation study datasets and convert them to YOLO format.")
    parser.add_argument("main_dataset_path", help="Path to the main dataset folder.")
    parser.add_argument("num_chunks", type=int, help="Number of chunks for ablation studies.")
    parser.add_argument("output_folder", help="Path to the output folder for ablation datasets.")
    parser.add_argument("test_dataset_path", help="Path to the test dataset folder.")
    parser.add_argument("--mode", choices=["detection", "segmentation", "pose_detection"], default="detection",
                        help="Processing mode for YOLO conversion ('detection', 'segmentation', 'pose_detection').")
    parser.add_argument("--classes", nargs='+', default=[], help="List of class names to process (default: all classes)")

    args = parser.parse_args()

    create_ablation_datasets(
        main_dataset_path=args.main_dataset_path,
        num_chunks=args.num_chunks,
        output_folder=args.output_folder,
        test_dataset_path=args.test_dataset_path,
        mode=args.mode,
        classes=args.classes
    )
