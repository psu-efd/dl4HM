from datetime import datetime


def print_now_time(string_before=" "):
    """Print the current time and date with an optional string in front

    :param

    :return:
    """

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("%s Date and time = %s" % (string_before, dt_string))
