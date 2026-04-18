from project.args import parse_args
from project.crop import run_cropping
from project.record import run_recording
from project.model import run_training, run_test


def run():
    args = parse_args()

    match args.command:
        case "train":
            run_training()
        case "record":
            run_recording(args.midi_name, args.camera)
        case "test":
            run_test()
        case "crop":
            run_cropping(args.path)

