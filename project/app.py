from project.args import parse_args
from project.dataset import build_dataset, dataset_info
from project.model import run_training, run_test

def run():
    args = parse_args()

    match args.command:
        case "train":
            run_training(args.train_dataset, args.test_dataset, args.batch_size, args.test_frequency, args.epochs)
        case "test":
            run_test()
        case "dataset":
            match args.dataset_command:
                case "build":
                    build_dataset(args.images, args.output)
                case "info":
                    dataset_info(args.path)
                case "record":
                    from project.record import run_recording
                    run_recording(args.midi_name, args.camera)
        case "crop":
            from project.crop import run_cropping
            run_cropping(args.path)
        case "vision":
            from project.vision import run_vision
            run_vision(args.model, args.camera)
