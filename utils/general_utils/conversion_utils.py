def convert_seconds_to_24h(seconds):
    hours = seconds // 60 // 60
    minutes = seconds // 60 % 60
    seconds = seconds % 60
    return hours, minutes, seconds