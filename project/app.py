from project.args import parse_args
from project.dataset import build_dataset, dataset_info
from project.model import run_training, run_test

def run():
    args = parse_args()

    match args.command:
        case "train":
            run_training()
        case "test":
            run_test()
        case "dataset":
            if args.dataset_command == "build":
                build_dataset(args.images, args.output)
            elif args.dataset_command == "info":
                dataset_info(args.path)
        case "record":
            from project.record import run_recording
            run_recording(args.midi_name, args.camera)
        case "crop":
            from project.crop import run_cropping
            run_cropping(args.path)
