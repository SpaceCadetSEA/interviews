class Rule:
    """
    Combination of condition/action
    """
    def __init__(self, condition, action) -> None:
        self.condition = condition
        self.action = action


class Condition:
    """
    if condition has functional aspect
    fx(field, value) => bool
    
    if condition eval to true, perform action
    
    operator logic lives here
    """
    def __init__(self, field, operator, value) -> None:
        self.field = field
        self.operator_func = operator
        self.value = value
        
    def evaluate(self, data) -> bool:
        if self.field not in data:
            raise ValueError()
        match data.keys():
            case self.field:
                
        

class RulesEngine:
    """
    list of rules <condition, action> pairs
    
    add_rule()
    evaluate()
    """
    def __init__(self) -> None:
        self.rules = set()
    
    def add_rule(self, condition, action):
        if action not in self.rules:
            # create new rule
            rule = Rule(condition, action)
            self.rules.add(rule)
        # TODO(atheis4): implement logic for duplicated rules or contrary rules
        
    def evaluate(self, data):
        actions = set()
        
        

# condition is just JSON for now
#  condition contains
#   field, operator, value
#  for part 2 and more advanced conditions, recurse

# input data in some form
