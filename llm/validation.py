def verify_input_data(input_data: dict, check_keys: list) -> bool:
    for key in check_keys:
        if key not in input_data:
            return False
    return True
