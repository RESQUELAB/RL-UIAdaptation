class Environment:
    '''
        * location = [indoor, outdoor]
    '''

    def __init__(self, config, atributes, combinations):
        self.config = config
        self.attributes = {}
        for aspect_name in atributes:
            value = atributes[aspect_name]
            setattr(self, aspect_name.lower(), value)
            self.attributes[aspect_name.lower()] = value
        self.combinations = combinations

    def __str__(self):
        environment_str = "Environment:\n"
        for aspect_name, aspect_value in self.attributes.items():
            environment_str += f"\t{aspect_name}: {aspect_value}\n"

        return environment_str

    def get_state(self):
        state = {}
        # Add attributes from 'attributes'
        for aspect_name, aspect_value in self.attributes.items():
            aspect_name_upper = aspect_name.upper()
            value = (self.config.environment_enums[aspect_name_upper][aspect_value].value) - 1
            state[aspect_name] = value
        return {"environment": state}

    def info(self):
        print(self)
