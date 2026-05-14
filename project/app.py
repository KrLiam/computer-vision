from project.args import parse_args

def run():
    args = parse_args()

    match args.command:
        case "train":
            from project.model import run_training
            run_training(
                args.train_dataset,
                args.test_dataset,
                args.batch_size,
                args.test_frequency,
                args.epochs,
                model_path=args.model,
                target_accuracy=args.target_accuracy,
            )
        case "test":
            from project.model import run_test
            run_test()
        case "dataset":
            from project.dataset import build_dataset, dataset_info
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
        case "area":
            from project.area import debug_keyboard
            debug_keyboard(args.input, args.threshold)
