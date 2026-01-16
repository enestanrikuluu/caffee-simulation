DRINK = "drink"
DRINK_AND_FOOD = "drink_and_food"

ALL_SERVICE_TYPES = [DRINK, DRINK_AND_FOOD]


def service_type_to_display_name(service_type):
    if service_type == DRINK:
        return "Drink"
    else:
        return "Drink + Food"
