class User:
    """
    Represents a user with attributes and preferences.

    Parameters:
    - attributes (dict): A dictionary containing user attributes.
    - preferences (dict): A dictionary containing user preferences.

    Example:
    >>> config = Config()
    >>> random_user = User(config.get_random_user(), {"layout": "grid", "theme": "light"})
    >>> print(random_user)
    User(attributes={'age': 'young', 'gender': 'female', 
                    'emotions': 'frustrated', 'experience': 'basic'}, 
                    preferences={'layout': 'grid', 'theme': 'light'})
    """

    def __init__(self, config, atributes, preferences, combinations):
        self.config = config
        self.attributes = {}
        for aspect_name in atributes:
            value = atributes[aspect_name]
            setattr(self, aspect_name.lower(), value)
            self.attributes[aspect_name.lower()] = value
        if hasattr(config, "user_preferences"):
            self.preferences = config.user_preferences
        else:
            self.preferences = preferences
        self.combinations = combinations

    def __str__(self):
        user_str = "User:\n"
        for aspect_name, aspect_value in self.attributes.items():
            user_str += f"\t{aspect_name}: {aspect_value}\n"

        user_str += "\tPreferences:\n"
        for pref_name, pref_value in self.preferences.items():
            user_str += f"\t\t{pref_name}: {pref_value}\n"
        return user_str

    def get_state(self):
        state = {}
        state['preferences'] = {}
        for pref_name, pref_value in self.preferences.items():
            pref_name_upper = pref_name.upper()
            value = (self.config.uidesign_enums[pref_name_upper][pref_value].value) - 1
            state["preferences"][pref_name] = value
        # Add attributes from 'attributes'
        for aspect_name, aspect_value in self.attributes.items():
            aspect_name_upper = aspect_name.upper()
            value = (self.config.user_enums[aspect_name_upper][aspect_value].value) - 1
            state[aspect_name] = value
        # Add preferences from 'preferences'
        
        return {"user": state}

    def info(self):
        print(self)

    def update_emotion(self):
        pass

    def compute_satisfaction(self, uidesign):
        satisfaction = 0

        for aspect, preference in self.preferences.items():
            if hasattr(uidesign, aspect):
                actual_value = getattr(uidesign, aspect)
                if actual_value == preference:
                    satisfaction += 1
        if satisfaction >= len(self.preferences.items()):
            satisfaction = 10
        return satisfaction
