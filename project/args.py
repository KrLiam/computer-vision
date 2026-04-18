import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Piano Vision Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train subcommand
    subparsers.add_parser("train", help="Train the neural network model")

    # Record subcommand
    record_parser = subparsers.add_parser("record", help="Record MIDI and video frames")
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

    # Crop subcommand
    crop_parser = subparsers.add_parser("crop", help="Crop dataset images")
    crop_parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        help="Glob patterns for image files to crop",
    )


    # Test command
    subparsers.add_parser("test", help="Run model inference on a random test sample")

    return parser.parse_args()
