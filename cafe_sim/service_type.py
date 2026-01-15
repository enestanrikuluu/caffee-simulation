from enum import Enum


class ServiceType(Enum):
    DRINK = "drink"
    DRINK_AND_FOOD = "drink_and_food"

    def to_display_name(self) -> str:
        if self == ServiceType.DRINK:
            return "Drink"
        else:
            return "Drink + Food"
