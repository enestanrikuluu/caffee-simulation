from enum import Enum


class PeriodKey(Enum):
    MONDAY_MORNING = "monday_morning"
    MONDAY_AFTERNOON = "monday_afternoon"
    MONDAY_EVENING = "monday_evening"
    TUESDAY_MORNING = "tuesday_morning"
    TUESDAY_AFTERNOON = "tuesday_afternoon"
    TUESDAY_EVENING = "tuesday_evening"

    @classmethod
    def from_string(cls, value: str) -> "PeriodKey":
        value_lower = value.lower().replace(" ", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"No PeriodKey matches '{value}'")

    def to_display_name(self) -> str:
        parts = self.value.split("_")
        return " ".join(word.capitalize() for word in parts)
