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
            from project.dataset import build_dataset, build_train_test_datasets, dataset_info
            match args.dataset_command:
                case "build":
                    if args.test_output:
                        build_train_test_datasets(
                            args.images,
                            train_output_path=args.output,
                            test_output_path=args.test_output,
                            test_ratio=args.test_ratio,
                            seed=args.split_seed,
                            all_frames=args.all_frames,
                            cap_none=args.cap_none,
                        )
                    else:
                        build_dataset(
                            args.images,
                            args.output,
                            all_frames=args.all_frames,
                            cap_none=args.cap_none,
                            seed=args.split_seed,
                        )
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
