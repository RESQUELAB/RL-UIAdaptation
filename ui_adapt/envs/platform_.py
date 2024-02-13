class Platform:
    def __init__(self, config, atributes, combinations):
        self.config = config
        self.attributes = {}
        for aspect_name in atributes:
            value = atributes[aspect_name]
            setattr(self, aspect_name.lower(), value)
            self.attributes[aspect_name.lower()] = value
        self.combinations = combinations

    def __str__(self):
        platform_str = "Platform:\n"
        for aspect_name, aspect_value in self.attributes.items():
            platform_str += f"\t{aspect_name}: {aspect_value}\n"

        return platform_str

    def get_state(self):
        state = {}
        for aspect_name, aspect_value in self.attributes.items():
            aspect_name_upper = aspect_name.upper()
            value = (self.config.platform_enums[aspect_name_upper][aspect_value].value) - 1
            state[aspect_name] = value
        return {"platform": state}

    def info(self):
        print(self)
