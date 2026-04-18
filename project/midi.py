def format_note(note: int):
    n = (note // 12) - 1
    match note % 12:
        case 0:
            return f"C{n}"
        case 1:
            return f"C#{n}"
        case 2:
            return f"D{n}"
        case 3:
            return f"D#{n}"
        case 4:
            return f"E{n}"
        case 5:
            return f"F{n}"
        case 6:
            return f"F#{n}"
        case 7:
            return f"G{n}"
        case 8:
            return f"G#{n}"
        case 9:
            return f"A{n}"
        case 10:
            return f"A#{n}"
        case 11:
            return f"B{n}"
