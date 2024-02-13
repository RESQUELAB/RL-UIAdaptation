import requests

class UIDesign:
    """
    Represents a User Interface design with its attributes.

    Parameters:
    - attributes (dict): A dictionary containing user attributes.

    Example:
    >>> config = Config()
    >>> random_ui = UIDesign(utils.get_random_ui(config))
    >>> print(random_ui)
    """

    def __init__(self, config, attributes, combinations):
        self.config = config
        self.mode = self.config.actions["MODE"]
        if "API" in self.mode and self.config.api_connection:
            host_name = self.config.api_connection["HOST"]
            port = self.config.api_connection["PORT"]
            self.url = host_name + ":" + str(port)
            self.parameters = {'change': ''}
            if self.config.api_connection["RENDER_RESOURCE"]:
                self.render_resource = self.config.api_connection["RENDER_RESOURCE"]
        self.attributes = {}
        for aspect_name in attributes:
            value = attributes[aspect_name]
            setattr(self, aspect_name.lower(), value)
            self.attributes[aspect_name.lower()] = value
        self.combinations = combinations

    def __str__(self):
        uidesign_str = "UIDesign:\n"
        for aspect_name, aspect_value in self.attributes.items():
            uidesign_str += f"\t{aspect_name}: {aspect_value}\n"

        return uidesign_str

    def update(self, target, value, api_call= ""):
        # Dynamically set the attribute in the UIDesign class based on the target
        target = target.lower()
        value = value.lower()
        err_msg = (
            f"{target.lower()!r} is not an attribute, use: "
            f"{self.attributes}. Check Config file."
        )
        assert hasattr(self, target), err_msg
        possible_values = [elem.name for elem in self.config.uidesign_enums[target.upper()]]
        err_msg = f"{value!r} is not a valid value, use: {possible_values}. Check Config file."
        assert value in possible_values, err_msg
        if self.mode == "API":
            assert api_call, "Api call not defined"
            api_call_split = api_call.split(" ")
            assert len(api_call_split) == 2, "api_call shoud have 2 parameters '<Resource> <Value>'"
            # Handle API request to update UI
            api_resource = api_call_split[0]
            api_value    = api_call_split[1]
            api_response = self.make_api_call(api_resource, api_value)
            if api_response != "success":
                err_msg = f"API request failed: {api_response}"
                return err_msg
        setattr(self, target, value)
        self.attributes[target] = value

    def make_api_call(self, resource, value):
        full_url = self.url+resource
        self.parameters["change"] = value
        response = requests.get(url = full_url, params = self.parameters)
        return response

    def picture(self, mode=""):
        if mode != "API":
            return
        response = self.make_api_call(self.render_resource, '')
        return response.json()

    def render(self, render_mode='ansii'):
        print(self)

    def get_state(self):
        state = {}
        # Add attributes from 'attributes'
        for aspect_name, aspect_value in self.attributes.items():
            aspect_name_upper = aspect_name.upper()
            value = (self.config.uidesign_enums[aspect_name_upper][aspect_value].value) - 1 
            state[aspect_name] = value
        return {"uidesign": state}
    