MONDAY_MORNING = "monday_morning"
MONDAY_AFTERNOON = "monday_afternoon"
MONDAY_EVENING = "monday_evening"
TUESDAY_MORNING = "tuesday_morning"
TUESDAY_AFTERNOON = "tuesday_afternoon"
TUESDAY_EVENING = "tuesday_evening"

ALL_PERIOD_KEYS = [
    MONDAY_MORNING,
    MONDAY_AFTERNOON,
    MONDAY_EVENING,
    TUESDAY_MORNING,
    TUESDAY_AFTERNOON,
    TUESDAY_EVENING,
]


def period_key_from_string(value):
    value_lower = value.lower().replace(" ", "_")
    if value_lower in ALL_PERIOD_KEYS:
        return value_lower
    raise ValueError(f"No PeriodKey matches '{value}'")


def period_key_to_display_name(period_key):
    parts = period_key.split("_")
    return " ".join(word.capitalize() for word in parts)
