import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Piano Vision Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the neural network model")
    train_parser.add_argument(
        "--train-dataset",
        type=str,
        default="dataset.tar",
        help="Path to the training dataset '.tar' or '.pt' file",
    )
    train_parser.add_argument(
        "--test-dataset",
        type=str,
        default="dataset.tar",
        help="Path to the test dataset '.tar' or '.pt' file",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    train_parser.add_argument(
        "--test-frequency",
        type=float,
        default=1.0,
        help="Frequency of testing during training (Example: 0.5 for every 2 epochs, 1.0 for every epoch)",
    )
    train_parser.add_argument(
        "--target-accuracy",
        type=float,
        default=1.0,
        help="Immediately stops training when tested accuracy reaches this value.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Path to load/save the '.pth' model file",
    )

    # Crop subcommand
    crop_parser = subparsers.add_parser("crop", help="Crop dataset images")
    crop_parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        help="Glob patterns for image files to crop",
    )

    # Vision subcommand
    vision_parser = subparsers.add_parser("vision", help="The real-time vision interface")
    vision_parser.add_argument(
        "--model",
        type=str,
        help="Path for the trained model to use.",
    )
    vision_parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Path to the video input device (Example: /dev/video3)",
    )

    # Area subcommand
    area_parser = subparsers.add_parser("area", help="Keyboard are debugging.")
    area_parser.add_argument(
        "--input",
        type=str,
        help="Path for the image.",
    )
    area_parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Whether should produce a single plot with the images.",
    )
    area_parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Whether the output images should be saved.",
    )

    # Dataset subcommand
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", help="Dataset commands")

    record_parser = dataset_subparsers.add_parser("record", help="Record MIDI and video frames")
    record_parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Path to the video input device (Example: /dev/video3)",
    )
    record_parser.add_argument(
        "--midi-name",
        type=str,
        required=True,
        help="Substring of the MIDI device name to connect to (Example: Casio)",
    )

    dataset_build_parser = dataset_subparsers.add_parser("build", help="Build the dataset from images")
    dataset_build_parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Glob patterns for image files to include in the dataset",
    )
    dataset_build_parser.add_argument(
        "--output",
        type=str,
        default="dataset.tar",
        help="Output dataset path, a '.tar' file, or train dataset path when --test-output is used",
    )
    dataset_build_parser.add_argument(
        "--test-output",
        type=str,
        help="Optional test dataset path. When set, the matched images are split into train and test datasets",
    )
    dataset_build_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples to put in the test dataset when --test-output is used",
    )
    dataset_build_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used to split train and test datasets",
    )
    dataset_build_parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Use every matched image as a dataset sample instead of grouping triplets",
    )
    dataset_build_parser.add_argument(
        "--cap-none",
        action="store_true",
        help="Limit samples with no pressed notes to the count of the most frequent note",
    )

    dataset_split_parser = dataset_subparsers.add_parser(
        "split",
        help="Build train and test datasets from images, like the recording interface auto split",
    )
    dataset_split_parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Glob patterns for image files to include in the datasets",
    )
    dataset_split_parser.add_argument(
        "--train-output",
        type=str,
        default="datasets/6_train.tar",
        help="Output path for the training dataset",
    )
    dataset_split_parser.add_argument(
        "--test-output",
        type=str,
        default="datasets/6_test.tar",
        help="Output path for the test dataset",
    )
    dataset_split_parser.add_argument(
        "--test-ratio",
        type=float,
        default=20,
        help="Percentage or fraction of samples to put in the test dataset, e.g. 20 or 0.2",
    )
    dataset_split_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used to split train and test datasets",
    )
    dataset_split_parser.add_argument(
        "--cap-none",
        action="store_true",
        default=True,
        help="Limit samples with no pressed notes to the count of the most frequent note",
    )
    dataset_split_parser.add_argument(
        "--no-cap-none",
        action="store_false",
        dest="cap_none",
        help="Do not limit samples with no pressed notes",
    )

    dataset_info_parser = dataset_subparsers.add_parser("info", help="Get dataset information")
    dataset_info_parser.add_argument(
        "--path",
        type=str,
        default="dataset.tar",
        help="Path to the dataset '.tar' or '.pt' file",
    )
    dataset_info_parser.add_argument(
        "--sort",
        action="store_true",
        default=False,
        help="Sort the note distribution by frequency",
    )
    # Test command
    test_parser = subparsers.add_parser("test", help="Run model inference on a random test sample")
    test_parser.add_argument(
        "--test-dataset",
        type=str,
        default="dataset.tar",
        help="Path to the test dataset '.tar' or '.pt' file",
    )
    test_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for testing",
    )
    test_parser.add_argument(
        "--model",
        type=str,
        help="Path for the trained model to use.",
    )

    return parser.parse_args()
