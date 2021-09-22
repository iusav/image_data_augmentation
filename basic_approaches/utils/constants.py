# Background
road_value = 7
ground_value = 6
sidewalk_value = 8
ground_values = [road_value, ground_value, sidewalk_value]
# Foreground
person_value = 24
aug_person_value = 50

# Occlusion
obstacle_values = [13, 14, 15, 17, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 50]


class Textcolor:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"